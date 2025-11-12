import torch
import torch.nn.functional as F
from wireframe_distillation.wireframe_net import WireframeNet

from ...settings import DATA_PATH
from ..base_model import BaseModel


def download_model(path):
    import subprocess

    if not path.parent.is_dir():
        path.parent.mkdir(parents=True, exist_ok=True)
    link = "https://raw.githubusercontent.com/iferfra/wireframe-detector/main/checkpoints/checkpoint.pth"
    cmd = ["wget", link, "-O", path]
    print("Downloading Wireframe model...")
    subprocess.run(cmd, check=True)


class WireframeSuarez(BaseModel):
    # Wireframe does not have way to define max num keypoints. Possibly as this would impede line detection possibility.
    default_conf = {
    }

    required_data_keys = ["image"]

    def _init(self, conf):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_name = "wireframe-suarez"

        ckpt = DATA_PATH / "weights" / self.model_name
        if not ckpt.is_file():
            download_model(ckpt)

        config = {"weights": ckpt, "size": [800, 800]}

        self.model = WireframeNet(config)
        self.model.eval().to(device)

    def _forward(self, data):
        results = self.model.forward(data)

        return {
            "lines": torch.tensor(results["line_segments"]).unsqueeze(0),
            "keypoints": results["points"],
            "keypoint_scores": results["scores"],
            "descriptors": results["descriptors"],
        }

    def sample_descriptors(self, torch_image, torch_points):
        """
        Performs forward pass to get descriptors for given points using DISK feature map.

        Args:
            torch_image: torch tensor [B, C, H, W], normalized image (grayscale or RGB)
            torch_points: torch tensor [B, N_b, 2], points in pixel coordinates

        Returns:
            list of tensors, one per batch image, each shaped [N_b, D]
        """
        b, c, h, w = torch_image.shape
        device = torch_image.device

        # Transform to model's expected size
        input_batch = self.model.transform_batch(torch_image)

        # Get dense feature map (DISK descriptors)
        with torch.no_grad():
            encoded_features = self.model.encoder(input_batch)
            decoded_features_disk = self.model.decoder_DISK(encoded_features)
            feature_map = decoded_features_disk[:, 1:129, :, :]  # Descriptor channels

        # Adjust points for resized image
        scale_x = self.model.size[1] / w
        scale_y = self.model.size[0] / h
        scaled_points = torch_points.clone()
        scaled_points[..., 0] *= scale_x
        scaled_points[..., 1] *= scale_y

        # Normalize to [-1, 1] for grid_sample
        img_wh = torch.tensor(
            [self.model.size[1] - 1, self.model.size[0] - 1], device=device
        )
        normalized_points = 2 * scaled_points / img_wh - 1

        # Sample using grid_sample
        descriptors_list = []
        for i in range(b):
            desc = F.grid_sample(
                feature_map[i : i + 1],
                normalized_points[i : i + 1, :, None, :],
                mode="bilinear",
                align_corners=True,
            )  # [1, D, N, 1]
            desc = desc.squeeze(3).squeeze(0).t()  # [N, D]
            desc = F.normalize(desc, p=2, dim=1)  # Normalize
            descriptors_list.append(desc)

        return descriptors_list

    def loss(self, pred, data):
        raise NotImplementedError

    def is_initialized(self):
        return True
