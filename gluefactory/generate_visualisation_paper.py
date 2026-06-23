"""
Minimal script to load line models and run them on HPatches images.

Author: Adapted from Paul-Edouard Sarlin (skydes)
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import torch

from gluefactory.eval.io import get_model, load_model
from gluefactory.settings import DATA_PATH
from gluefactory.visualization.viz2d import plot_images, plot_lines


def visualize_img_with_gt(output, input_image, save_path):
    plt.figure()
    _, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Right: detected lines
    plot_images([input_image], [], cmaps="gray")
    plot_lines([output["lines"][0]], indices=range(1))

    plt.savefig(save_path)
    plt.close()


def load_rgb_tensor(image_path):
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    return img_rgb, img_tensor


def get_hpatches_images(root, num_images, view):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"HPatches root does not exist: {root}")

    images = []
    for sequence in sorted(p for p in root.iterdir() if p.is_dir()):
        image_path = sequence / f"{view}.ppm"
        if image_path.exists():
            images.append(image_path)
        if num_images is not None and len(images) >= num_images:
            break

    if not images:
        raise ValueError(f"No HPatches '*{view}.ppm' images found under {root}")
    return images


def get_jpl_model():
    return load_model(
        {
            "name": "extractors.joint_point_line_extractor",
            "line_df_decoder_channels": 64,
            "training": {"do": False},
            "max_num_keypoints": 2048,
            "line_detection": {"do": True, "conf": {"lsd_type": "old", "sigma": 0.9}},
            "checkpoint": "assets/jpl_best.tar",
        },
        "./assets/jpl_best_with_points.tar",
    ).to("cpu")


def get_scalelsd_model():
    return get_model("lines.scalelsd")({"name": "lines.scalelsd", "threshold": 10}).to(
        "cpu"
    )


def get_wireframe_model():
    return get_model("lines.wireframe_suarez")({"name": "lines.wireframe_suarez"}).to(
        "cpu"
    )


def get_deeplsd_model():
    return get_model("lines.deeplsd")(
        {
            "name": "lines.deeplsd",
            " model_conf": {
                "detect_lines": True,
                "line_detection_params": {
                    "use_img_grad_angle": False,
                    "merge": False,
                    "grad_nfa": True,
                    "filtering": True,
                    "faster_lsd": False,
                },
            },
        }
    ).to("cpu")


def get_lsd_model():
    return get_model("lines.lsd")({"name": "lines.lsd"}).to("cpu")


# Command to run it: python -m gluefactory.generate_visualisation_paper
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hpatches-root",
        type=Path,
        default=DATA_PATH / "hpatches-sequences-release",
        help="Path to the HPatches sequences directory.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=5,
        help="Number of HPatches sequences to visualize. Use -1 for all.",
    )
    parser.add_argument(
        "--view",
        type=int,
        default=1,
        choices=range(1, 7),
        help="HPatches view index to visualize for each sequence.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("generated_jpl_images/hpatches"),
        help="Directory where visualizations are written.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    num_images = None if args.num_images < 0 else args.num_images
    image_paths = get_hpatches_images(args.hpatches_root, num_images, args.view)

    # Visualize results
    models = [
        (get_lsd_model, "lsd"),
        (get_deeplsd_model, "deeplsd"),
        (get_jpl_model, "jpl"),
        (get_scalelsd_model, "scalelsd"),
        (get_wireframe_model, "wireframe"),
    ]

    for getter, name in models:
        print(f"Loading {name}...")
        model = getter()
        model.eval()

        for image_path in image_paths:
            img_rgb, img_tensor = load_rgb_tensor(image_path)
            with torch.no_grad():
                output = model({"image": img_tensor})

            sequence = image_path.parent.name
            path = args.output_dir / f"{sequence}_{image_path.stem}_{name}.pdf"
            visualize_img_with_gt(output, img_rgb, path)
            print(f"Visualization saved to {path}")
