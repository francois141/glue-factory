"""
Visualization script for JPL outputs.

Author: Adapted from Paul-Edouard Sarlin (skydes)
"""

import argparse
import copy
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from omegaconf import OmegaConf
from gluefactory.eval.io import load_model
from .datasets import get_dataset


def nicer_display(df):
    return (df) ** (1/4)


# Command:
#  python -m gluefactory.visualise_intermediate_layer_jpl --conf gluefactory/configs/visualise_jpl_output.yaml
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, default=None)
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_intermixed_args()

    conf = OmegaConf.from_cli(args.dotlist)
    if args.conf:
        conf = OmegaConf.merge(OmegaConf.load(args.conf), conf)

    # Prepare dataset
    data_conf = copy.deepcopy(conf.data)
    dataset = get_dataset(data_conf.name)(data_conf)
    train_loader = dataset.get_dataset("train")
    print("Train dataset loaded")

    # Load model
    print("Loading model")
    jpldd_model = load_model(conf.model.extractor, "./assets/jpl_best_with_points.tar").to("cpu")
    jpldd_model.eval()
    print("Model loaded")

    # Output folder
    out_dir = Path("generated_jpl_images")
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_fig(img, filename, cmap=None):
        plt.imshow(img, cmap=cmap)
        plt.axis("off")
        plt.savefig(out_dir / f"{filename}.pdf", dpi=300, bbox_inches="tight")
        plt.close()

    for it, data in enumerate(train_loader):
        if it < 9:
            continue
        if it > 9:
            break

        print(f"Generating visualizations for iteration {it}")
        output_jpldd = jpldd_model({"image": data["image"].to("cpu").unsqueeze(0)})

        # ===============================
        # Image visualization
        # ===============================
        save_fig(data["image"].cpu().numpy().transpose(1, 2, 0), f"image_{it}")

        # ===============================
        # Backbone PCA visualization
        # ===============================
        x = output_jpldd["backbone"].squeeze(0)  # [C,H,W]
        features = x.permute(1, 2, 0).reshape(-1, x.shape[0]).cpu().numpy()  # [H*W,C]

        pca = PCA(n_components=3)
        features_pca = pca.fit_transform(features)
        features_pca -= features_pca.min()
        features_pca /= features_pca.max()
        img = features_pca.reshape(x.shape[1], x.shape[2], 3)
        save_fig(img, f"pca_features_{it}")

        # ===============================
        # Line distance field
        # ===============================
        img_np = output_jpldd["line_distancefield"].detach().cpu().numpy()[0]
        img_vis = -nicer_display(img_np)
        save_fig(img_vis, f"line_distance_field_{it}")

        # ===============================
        # Keypoint + junction score map
        # ===============================
        kpt_map = output_jpldd["keypoint_and_junction_score_map"].detach().cpu().numpy()
        if kpt_map.ndim == 3:
            for c in range(kpt_map.shape[0]):
                save_fig(kpt_map[c], f"kp_junction_map_{it}_ch{c}", cmap="viridis")
        else:
            save_fig(kpt_map, f"kp_junction_map_{it}", cmap="viridis")
