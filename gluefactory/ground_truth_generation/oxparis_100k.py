"""
Run the homography adaptation with Superpoint for 100k images from the oxford paris revisited dataset
Goal: create groundtruth with superpoint. Format: stores groundtruth for every image in a separate file.
"""

import argparse
import random
from pathlib import Path

# TODO: Install this library
import afm_op
import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from faster_pytlsd import lsd as fast_lsd
from joblib import Parallel, delayed
from kornia.geometry.transform import warp_perspective
from kornia.morphology import erosion
from omegaconf import OmegaConf
from PIL import Image
from pytlsd import lsd
from scipy.ndimage import maximum_filter
from tqdm import tqdm

from gluefactory.datasets import get_dataset
from gluefactory.datasets.homographies_deeplsd import sample_homography_deeplsd
from gluefactory.geometry.homography import sample_homography_corners
from gluefactory.ground_truth_generation.generate_gt_deeplsd import (
    generate_ground_truth_with_homography_adaptation,
)
from gluefactory.models.extractors.dad import DadDetector
from gluefactory.models.extractors.superpoint import top_k_keypoints
from gluefactory.models.extractors.superpoint_open import SuperPoint
from gluefactory.models.lines.deeplsd import DeepLSD
from gluefactory.settings import DATA_PATH, EVAL_PATH


class KPExtractor:
    def __init__(self, config):
        self.threshold_type = config.get("threshold_type", "nms")
        self.threshold_value = config.get("threshold_value", 0.015)
        self.max_keypoints = config.get(
            "max_keypoints", None
        )  # Default to None if not provided

    def extract_keypoints(self, sp_heatmap):
        nms = (sp_heatmap == maximum_filter(sp_heatmap, size=3)) & (
            sp_heatmap > self.threshold_value
        )
        keypoints = np.argwhere(nms)  # (row, col)
        scores = sp_heatmap[nms]  # Extract scores

        if self.max_keypoints is not None and len(keypoints) > self.max_keypoints:
            idx = np.argsort(scores)[::-1][: self.max_keypoints]
            keypoints = keypoints[idx]

        return keypoints


class KPExtractor:
    @staticmethod
    def simple_nms(scores, radius):
        """Perform non maximum suppression on the heatmap using max-pooling.
        This method does not suppress contiguous points that have the same score.
        Args:
            scores: the score heatmap of size `(B, H, W)`.
            size: an interger scalar, the radius of the NMS window.
        """

        def max_pool(x):
            return torch.nn.functional.max_pool2d(
                x, kernel_size=radius * 2 + 1, stride=1, padding=radius
            )

        zeros = torch.zeros_like(scores)
        max_mask = scores == max_pool(scores)
        for _ in range(2):
            supp_mask = max_pool(max_mask.float()) > 0
            supp_scores = torch.where(supp_mask, zeros, scores)
            new_max_mask = supp_scores == max_pool(supp_scores)
            max_mask = max_mask | (new_max_mask & (~supp_mask))
        return torch.where(max_mask, scores, zeros)

    @staticmethod
    def remove_borders(keypoints, b, h, w):
        mask_h = (keypoints[1] >= b) & (keypoints[1] < (h - b))
        mask_w = (keypoints[2] >= b) & (keypoints[2] < (w - b))
        mask = mask_h & mask_w
        return (keypoints[0][mask], keypoints[1][mask], keypoints[2][mask])

    @staticmethod
    def soft_argmax_refinement(keypoints, scores, radius: int):
        width = 2 * radius + 1
        sum_ = torch.nn.functional.avg_pool2d(
            scores[:, None], width, 1, radius, divisor_override=1
        )
        sum_ = torch.clamp(sum_, min=1e-6)
        ar = torch.arange(-radius, radius + 1).to(scores)
        kernel_x = ar[None].expand(width, -1)[None, None]
        dx = torch.nn.functional.conv2d(scores[:, None], kernel_x, padding=radius)
        dy = torch.nn.functional.conv2d(
            scores[:, None], kernel_x.transpose(2, 3), padding=radius
        )
        dydx = torch.stack([dy[:, 0], dx[:, 0]], -1) / sum_[:, 0, :, :, None]
        refined_keypoints = []
        for i, kpts in enumerate(keypoints):
            delta = dydx[i][tuple(kpts.t())]
            refined_keypoints.append(kpts.float() + delta)
        return refined_keypoints

    def __init__(self, config):
        self.threshold_value = config.get("threshold_value", 0.015)
        self.max_num_keypoints = config.get("max_keypoints", 8000)
        self.nms_radius = config.get("nms_radius", 4)
        self.remove_borders = config.get("remove_borders", 4)
        self.refinement_radius = config.get("refinement_radius", 0)

    def extract_keypoints(self, sp_heatmap, b_size=1):
        """Extract keypoints as implemented in Superpoint"""
        h, w = sp_heatmap.shape[1] // 8, sp_heatmap.shape[2] // 8

        scores = self.simple_nms(sp_heatmap, self.nms_radius)

        # Extract keypoints
        best_kp = torch.where(scores > self.threshold_value)

        # Discard keypoints near the image borders
        best_kp = self.remove_borders(best_kp, self.remove_borders, h * 8, w * 8)
        scores = scores[best_kp]

        # Separate into batches
        keypoints = [
            torch.stack(best_kp[1:3], dim=-1)[best_kp[0] == i] for i in range(b_size)
        ]
        scores = [scores[best_kp[0] == i] for i in range(b_size)]

        # Keep the k keypoints with highest score
        if self.max_num_keypoints > 0:
            keypoints, scores = list(
                zip(
                    *[
                        top_k_keypoints(k, s, self.max_num_keypoints)
                        for k, s in zip(keypoints, scores)
                    ]
                )
            )
            keypoints, scores = list(keypoints), list(scores)

        if self.refinement_radius > 0:
            keypoints = self.soft_argmax_refinement(
                keypoints, sp_heatmap, self.refinement_radius
            )

        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]
        return keypoints


conf = {
    "patch_shape": [800, 800],
    "difficulty": 0.8,
    "translation": 1.0,
    "n_angles": 10,
    "max_angle": 60,
    "min_convexity": 0.05,
}

dad_conf = {
    "max_num_keypoints": 1024,
}

sp_conf = {
    "max_num_keypoints": None,
    "nms_radius": 4,
    "detection_threshold": 0.005,
    "remove_borders": 4,
    "descriptor_dim": 256,
    "channels": [64, 64, 128, 128, 256],
    "dense_outputs": None,
    "weights": None,  # local path of pretrained weights
}

H_params = {
    "difficulty": 0.8,
    "translation": 1.0,
    "max_angle": 60,
    "n_angles": 10,
    "min_convexity": 0.05,
}

ha = {
    "enable": False,
    "num_H": 100,
    "mini_bs": 1,
    "aggregation": "mean",
}

homography_params = {
    "translation": True,
    "rotation": True,
    "scaling": True,
    "perspective": True,
    "scaling_amplitude": 0.2,
    "perspective_amplitude_x": 0.2,
    "perspective_amplitude_y": 0.2,
    "patch_ratio": 0.85,
    "max_angle": 1.57,
    "allow_artifacts": True,
}


def warp_points(points: torch.Tensor, H) -> torch.Tensor:
    """Warp 2D points by a homography H using PyTorch tensors."""
    H = torch.tensor(H).to(device)
    n_points = points.shape[0]

    # Swap x and y (axis 1 becomes axis 0)
    reproj_points = points.clone()[:, [1, 0]]

    # Add homogeneous coordinate (1s)
    ones = torch.ones((n_points, 1), dtype=points.dtype, device=points.device)
    reproj_points = torch.cat([reproj_points, ones], dim=1)  # shape (N, 3)

    # Apply homography
    reproj_points = (H.to(torch.float32) @ reproj_points.T).T  # shape (N, 3)

    # Convert back from homogeneous coordinates
    reproj_points = reproj_points[:, :2] / reproj_points[:, 2:].clamp(min=1e-8)

    # Swap back to original x, y order
    reproj_points = reproj_points[:, [1, 0]]

    return reproj_points


def warp_lines(lines, H):
    """Warp lines of the shape [N, 2, 2] by an homography H."""
    return warp_points(lines.reshape(-1, 2), H).reshape(-1, 2, 2)


# Deep LSD Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conf = {
    "detect_lines": False,  # Whether to detect lines or only DF/AF
    "line_detection_params": {
        "merge": False,  # Whether to merge close-by lines
        "filtering": True,
        # Whether to filter out lines based on the DF/AF. Use 'strict' to get an even stricter filtering
        "grad_thresh": 3,
        "grad_nfa": True,
        # If True, use the image gradient and the NFA score of LSD to further threshold lines. We recommand using it for easy images, but to turn it off for challenging images (e.g. night, foggy, blurry images)
    },
}

# Load the model
deeplsd_network = DeepLSD(conf)
deeplsd_network = deeplsd_network.to(device).eval()


def ha_df_deeplsd(path, img, is_analysis, num=100, border_margin=3, min_counts=5):
    """Perform homography adaptation to regress line distance function maps.
    Args:
        img: a grayscale np image.
        num: number of homographies used during HA.
        border_margin: margin used to erode the boundaries of the mask.
        min_counts: any pixel which is not activated by more than min_count is BG.
    Returns:
        The aggregated distance function maps in pixels
        and the angle to the closest line.
    """
    h, w = img.shape[:2]
    size = (w, h)
    df_maps = []
    angles = []
    counts = []

    # We currently set to a single homography only
    num = 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (border_margin * 2, border_margin * 2)
    )

    deeplsd_generation = True
    if deeplsd_generation:
        df, angle = generate_ground_truth_with_homography_adaptation(
            torch.tensor(img).to(device).unsqueeze(0), deeplsd_network
        )
        return df.cpu().numpy(), angle.cpu().numpy()

    with torch.no_grad():
        pix_loc = torch.tensor(
            np.stack(np.meshgrid(np.arange(h), np.arange(w), indexing="ij"), axis=-1)
        ).to(device)

        previous_value = None
        analysis_values_mean = []
        analysis_values_max = []

        raster_lines = np.zeros_like(img[:, :, 0])

        # Loop through all the homographies
        for i in range(num):
            # Generate a random homography
            if i == 0:
                H = np.eye(3)
            else:
                H = sample_homography_deeplsd(img.shape[:2], **homography_params)
            H_inv = np.linalg.inv(H)

            # Warp the image
            warped_img = cv2.warpPerspective(
                img, H, size, borderMode=cv2.BORDER_REPLICATE
            )

            # Regress the DF on the warped image
            warped_lines = lsd((warped_img * 255).astype(np.uint8))[
                :, [1, 0, 3, 2]
            ].reshape(-1, 2, 2)

            # Warp the lines back
            warped_lines = torch.tensor(warped_lines).to(device)
            lines = warp_lines(warped_lines, H_inv)

            # Get the DF and angles
            num_lines = len(lines)
            cuda_lines = lines[:, :, [1, 0]].to(torch.float32)
            cuda_lines = cuda_lines.reshape(-1, 4)[None].cuda()
            offset = afm_op.afm(
                cuda_lines, torch.IntTensor([[0, num_lines, h, w]]).cuda(), h, w
            )[0]

            offset = offset[0].permute(1, 2, 0)[:, :, [1, 0]]

            df = torch.norm(offset, dim=-1)
            angle = torch.remainder(
                torch.arctan2(offset[:, :, 0], offset[:, :, 1]) + torch.pi / 2, torch.pi
            )

            # Compute the valid pixels
            count = cv2.warpPerspective(
                np.ones_like(img), H_inv, size, flags=cv2.INTER_NEAREST
            )
            count = cv2.erode(count, kernel)
            counts.append(count)

            df_maps.append(df)
            angles.append(angle)

            raster_lines += (df < 1).cpu().numpy().astype(np.uint8) * count

        # Compute the median of all DF maps, with counts as weights
        df_maps = torch.stack(df_maps)
        angles = torch.stack(angles)
        counts = torch.tensor(np.stack(counts)).to(device)

        # Median of the DF
        df_maps[counts == 0] = torch.nan
        avg_df = torch.nanmedian(df_maps, dim=0).values.cpu().numpy()

        # Median of the angle
        circ_bound = (torch.minimum(torch.pi - angles, angles) * counts).sum(
            0
        ) / counts.sum(0) < 0.3
        angles[:, circ_bound] -= torch.where(
            angles[:, circ_bound] > torch.pi / 2,
            torch.ones_like(angles[:, circ_bound]) * torch.pi,
            torch.zeros_like(angles[:, circ_bound]),
        )
        angles[counts == 0] = torch.nan
        avg_angle = (
            torch.remainder(torch.nanmedian(angles, axis=0).values, torch.pi)
            .cpu()
            .numpy()
        )

        # Generate the background mask and a saliency score
        raster_lines = np.where(
            raster_lines > min_counts,
            np.ones_like(img[:, :, 0]),
            np.zeros_like(img[:, :, 0]),
        )
        raster_lines = cv2.dilate(raster_lines, np.ones((21, 21), dtype=np.uint8))
        bg_mask = (1 - raster_lines).astype(float)
        return avg_df, avg_angle, bg_mask


def get_dataset_and_loader(
    num_workers: int, dataset: str, chunk: int
):  # folder where dataset images are placed
    print("Loading Dataset {}...".format(dataset))
    config = {
        "name": dataset,  # name of dataset class in gluefactory > datasets
        "grayscale": True,  # commented out things -> dataset must also have these keys but has not
        "train_batch_size": 1,  # prefix must match split mode
        "val_batch_size": 1,  # prefix must match split mode
        "all_batch_size": 1,
        "chunk": chunk,  # chunk of the dataset to use
        "num_workers": num_workers,
        "split": (
            "all" if dataset in ["minidepth", "scannet"] else "train"
        ),  # if implemented by dataset class gives different splits
    }
    omega_conf = OmegaConf.create(config)
    dataset = get_dataset(omega_conf.name)(omega_conf)
    loader = dataset.get_data_loader(omega_conf.get("split", "all"))
    return loader


def ha_forward_points(img, is_analysis, num=100):
    """Perform homography adaptation to regress line distance function maps.
    Args:
        img: a grayscale np image.
        num: number of homographies used during HA.
        border_margin: margin used to erode the boundaries of the mask.
        min_counts: any pixel which is not activated by more than min_count is BG.
    Returns:
        The aggregated distance function maps in pixels
        and the angle to the closest line.
    """
    h, w = img.shape[:2]

    num = 1

    img = np.transpose(img, (2, 0, 1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_dad = True

    if use_dad:
        model = DadDetector(sp_conf).to(device)
        model.eval().to(device)
    else:
        model = SuperPoint(sp_conf).to(device)
        model.eval().to(device)

    Hs = []
    for i in range(num):
        if i == 0:
            # Always include at least the identity
            Hs.append(torch.eye(3, dtype=torch.float, device=device))
        else:
            Hs.append(
                torch.tensor(
                    sample_homography_corners((w, h), patch_shape=(w, h), **H_params)[
                        0
                    ],
                    dtype=torch.float,
                    device=device,
                )
            )
    Hs = torch.stack(Hs, dim=0)

    bs = ha["mini_bs"]
    B = 1

    erosion_kernel = torch.tensor(
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=torch.float,
    )

    erosion_kernel = erosion_kernel.to(device)

    sp_image_tensor = torch.tensor(img, dtype=torch.float32, device=device).unsqueeze(0)
    n_mini_batch = int(np.ceil(num / bs))
    scores = torch.empty((B, 0, h, w), dtype=torch.float, device=device)
    counts = torch.empty((B, 0, h, w), dtype=torch.float, device=device)

    previous_value = None
    delta_max_diff = []

    for i in range(n_mini_batch):
        H = Hs[i * bs : (i + 1) * bs]
        nh = len(H)
        H = H.repeat(B, 1, 1)
        H = H.to(device)

        a = torch.repeat_interleave(sp_image_tensor, nh, dim=0)
        warped_imgs = warp_perspective(a, H, (h, w), mode="bilinear")

        for j, img in enumerate(warped_imgs):
            with torch.no_grad():
                img1 = img / 255.0  # Normalize image
                img1 = img1.unsqueeze(0)  # Add batch dimension
                pred = model({"image": img1.to(device)})
                pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

                warped_heatmap = pred["heatmap"]

                # convert to pytorch tensor
                score = torch.tensor(
                    warped_heatmap, dtype=torch.float32, device=device
                ).unsqueeze(0)

                # Compute valid pixels
                H_inv = torch.inverse(H[j])

                count = warp_perspective(
                    torch.ones_like(score).unsqueeze(1),
                    H[j].unsqueeze(0),
                    (h, w),
                    mode="nearest",
                )

                count = erosion(count, erosion_kernel)
                count = warp_perspective(
                    count, H_inv.unsqueeze(0), (h, w), mode="nearest"
                )[:, 0]
                score = warp_perspective(
                    score[:, None], H_inv.unsqueeze(0), (h, w), mode="bilinear"
                )[:, 0]

            scores = torch.cat([scores, score.reshape(B, 1, h, w)], dim=1)
            counts = torch.cat([counts, count.reshape(B, 1, h, w)], dim=1)
            scores[counts == 0] = 0
            score = scores.max(dim=1)[0]

            scoremap = score.squeeze(0)

    return scoremap


def check_and_save_base_image_if_not_exists(img_data, output_folder_path, image_u8):
    # First add folder
    complete_out_folder = (output_folder_path / str(img_data["name"][0])).parent
    complete_out_folder.mkdir(parents=True, exist_ok=True)

    base_image_path = (
        complete_out_folder
        / f"{Path(img_data['name'][0]).name.split('.')[0]}_base_image.jpg"
    )

    if not base_image_path.exists():
        # Save the base image as jpg
        Image.fromarray(image_u8).save(base_image_path, format="JPEG")


def process_points(img_data, num_H, output_folder_path, is_analysis):
    """
    Perform homography adaptation with superpoint for a given image and store results.
    """
    assert len(img_data["name"]) == 1  # Currently expect batch size one!

    complete_out_folder = (output_folder_path / str(img_data["name"][0])).parent
    points_image_path = (
        complete_out_folder
        / f"{Path(img_data['name'][0]).name.split('.')[0]}_heatmap.npy"
    )

    # If we already generated the value for this one, we can ignore it - that way we can transparently restart the workload later
    if points_image_path.exists():
        return

    image_numpy = np.transpose(img_data["image"].numpy()[0], (1, 2, 0))  # H x W x C

    # Then add image if it doesn't exist
    image_u8 = (image_numpy * 255).astype(np.uint8)
    check_and_save_base_image_if_not_exists(img_data, output_folder_path, image_u8)

    # Then generate the dataset with homography adapataion
    superpoint_heatmap = ha_forward_points(image_u8, is_analysis, num=num_H).cpu()
    np.save(points_image_path, superpoint_heatmap.cpu().numpy())


def process_distance_field(img_data, num_H, output_folder_path, is_analysis):
    """
    Perform homography adaptation with faster_lsd for a given image and store results.
    """
    assert len(img_data["name"]) == 1  # Currently expect batch size one!

    complete_out_folder = (output_folder_path / str(img_data["name"][0])).parent
    lines_image_path = (
        complete_out_folder / f"{Path(img_data['name'][0]).name.split('.')[0]}_df.npy"
    )

    # If we already generated the value for this one, we can ignore it - that way we can transparently restart the workload later
    if lines_image_path.exists():
        return

    image_numpy = np.transpose(img_data["image"].numpy()[0], (1, 2, 0))  # H x W x C

    # Then add image if it doesn't exist
    image_u8 = (image_numpy * 255).astype(np.uint8)
    check_and_save_base_image_if_not_exists(img_data, output_folder_path, image_u8)

    # Run homography adaptation
    df, af = ha_df_deeplsd(img_data["name"], image_numpy, is_analysis, num=num_H)

    # Save the distance field
    np.save(
        complete_out_folder / f"{Path(img_data['name'][0]).name.split('.')[0]}_df.npy",
        df,
    )

    # Save the angle field
    np.save(
        complete_out_folder / f"{Path(img_data['name'][0]).name.split('.')[0]}_af.npy",
        af,
    )


def process_both(img_data, num_H, output_folder_path, is_analysis):
    """
    Perform homography adaptation with faster_lsd and superpoint for a given image and store results.
    """
    process_distance_field(img_data, num_H, output_folder_path, is_analysis)
    process_points(img_data, num_H, output_folder_path, is_analysis)


def export_ha(data_loader, output_folder_path, num_H: int, n_jobs: int, type: str):
    if type == "points":
        Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(process_points)(img_data, num_H, output_folder_path, args.analysis)
            for img_data in tqdm(data_loader, total=len(data_loader))
        )
    elif type == "lines":
        Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(process_distance_field)(
                img_data, num_H, output_folder_path, args.analysis
            )
            for img_data in tqdm(data_loader, total=len(data_loader))
        )
    elif type == "both":
        Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(process_both)(img_data, num_H, output_folder_path, args.analysis)
            for img_data in tqdm(data_loader, total=len(data_loader))
        )
    else:
        print("Unknown generation type")


# Command to launch
# python -m gluefactory.ground_truth_generation.oxparis_100k oxford_paris_mini_100k --num_H 1
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset", choices=["minidepth", "oxford_paris_mini_100k", "scannet"]
    )
    parser.add_argument(
        "--type",
        type=str,
        help="Choose what to generate [points|lines]",
        choices=["points", "lines", "both"],
        default="both",
    )
    parser.add_argument(
        "--output_folder", type=str, help="Output folder.", default="oxparis_100k"
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=0,
        help="Chunk of the dataset to use. Default is 0, which is the first chunk.",
    )
    parser.add_argument(
        "--num_H", type=int, default=100, help="Number of homographies used during HA."
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of jobs (that perform HA) to run in parallel.",
    )
    parser.add_argument(
        "--n_jobs_dataloader",
        type=int,
        default=1,
        help="Number of jobs the dataloader uses to load images",
    )
    parser.add_argument(
        "--analysis",
        action="store_true",
        help="Analyse max difference between iterations",
    )
    args = parser.parse_args()

    out_folder_path = EVAL_PATH / args.output_folder
    out_folder_path.mkdir(exist_ok=True, parents=True)

    print("DATASET: ", args.dataset)
    print("OUTPUT PATH: ", out_folder_path)
    print("NUMBER OF HOMOGRAPHIES: ", args.num_H)
    print("N JOBS: ", args.n_jobs)
    print("N DATALOADER JOBS: ", args.n_jobs_dataloader)

    dataloader = get_dataset_and_loader(
        args.n_jobs_dataloader, args.dataset, args.chunk
    )
    export_ha(dataloader, out_folder_path, args.num_H, args.n_jobs, args.type)
    print("Done !")
