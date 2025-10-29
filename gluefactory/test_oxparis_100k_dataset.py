"""
A generic training script that works with any model and dataset.

Author: Paul-Edouard Sarlin (skydes)
"""

import argparse
import copy
import re
import shutil
import signal
from collections import defaultdict
from pathlib import Path
from pydoc import locate

import flow_vis
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.ndimage import binary_dilation
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from . import __module_name__, logger
from .datasets import get_dataset
from .eval import run_benchmark
from .models import get_model
from .settings import EVAL_PATH, TRAINING_PATH
from .utils.experiments import get_best_checkpoint, get_last_checkpoint, save_experiment
from .utils.stdout_capturing import capture_outputs
from .utils.tensor import batch_to_device
from .utils.tools import (
    AverageMetric,
    MedianMetric,
    PRMetric,
    RecallMetric,
    fork_rng,
    set_seed,
)


def nicer_display(df):
    return (df) ** (1 / 4)


def get_flow_vis(df, ang, line_neighborhood=5):
    norm = line_neighborhood + 1 - np.clip(df, 0, line_neighborhood)
    flow_uv = np.stack([norm * np.cos(ang), norm * np.sin(ang)], axis=-1)
    flow_img = flow_vis.flow_to_color(flow_uv, convert_to_bgr=False)
    return flow_img


def visualize_img_with_gt(name, dset, num=5, offset=0, lim_kpoints=-1):
    plt.figure()
    idxs = list(range(offset, offset + num, 1))
    _, ax = plt.subplots(1, 4, figsize=(20, 8))
    for i in idxs:
        if i >= 1:
            break
        elem = dset[i]

        hmap = elem["superpoint_heatmap"]
        df = elem["deeplsd_distance_field"]
        af = elem["deeplsd_angle_field"]

        ax[0].axis("off")
        ax[0].set_title(f"Heatmap ({hmap.shape})")
        ax[0].imshow(hmap)

        ax[1].axis("off")
        ax[1].set_title("Distance Field")
        ax[1].imshow(df)

        ax[2].axis("off")
        ax[2].set_title("Angle Field")
        ax[2].imshow(get_flow_vis(df, af))

        ax[3].axis("off")
        ax[3].set_title("Original")
        ax[3].imshow(elem["image"].permute(1, 2, 0))

    plt.savefig(f"test_{name}.png")
    plt.show()
    plt.close()


# Command to launch it
# python -m gluefactory.test_oxparis_100k_dataset --conf gluefactory/configs/train_jpl_oxparis_100k.yaml
# python -m gluefactory.test_oxparis_100k_dataset --conf gluefactory/configs/train_jpl_oxparis_base.yaml
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, default=None)
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_intermixed_args()

    conf = OmegaConf.from_cli(args.dotlist)
    if args.conf:
        conf = OmegaConf.merge(OmegaConf.load(args.conf), conf)

    data_conf = copy.deepcopy(conf.data)
    dataset = get_dataset(data_conf.name)(data_conf)

    train_loader = dataset.get_dataset("train")

    for it, data in enumerate(train_loader):
        print(f"=== Running iteration {it} ===")
        dataset = {}

        # Load image
        dataset["image"] = data["image"]

        # Load distance field
        img_np = data["deeplsd_distance_field"].numpy()
        img_np = nicer_display(img_np) * 255
        dataset["deeplsd_distance_field"] = -img_np

        # Load angle field
        img_np = data["deeplsd_angle_field"].numpy()
        dataset["deeplsd_angle_field"] = img_np

        # Load superpoint Heatmap
        img_np = data["superpoint_heatmap"].numpy()
        dataset["superpoint_heatmap"] = img_np * 255
        visualize_img_with_gt(it, [dataset])
