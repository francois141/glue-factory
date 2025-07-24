"""
Performs timing on a model and a dataset. Model and dataset must be specified in a conf file.
Attention: Make sure you configure batch_size and dataset split correctly.

- Ex run: python -m gluefactory.eval.timing_lsd_vs_faster_lsd
"""

import argparse
import time
import math

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from itertools import islice

from gluefactory.datasets import get_dataset
from gluefactory.models import get_model
from gluefactory.utils.tensor import batch_to_device

conf =  OmegaConf.create({
    "lsd_model": { "name": "lines.lsd", "faster_lsd": False},
    "fastlsd_model": {"name": "lines.lsd", "faster_lsd": True},
    "data_rdnim": {"name": "gluefactory.datasets.rdnim"},
    "data_hpatches": {"name": "gluefactory.datasets.hpatches"},
    "data_oxford_paris": {"name": "gluefactory.datasets.oxford_paris_mini_1view_jpldd"},
    "data_eth": {"name": "gluefactory.datasets.eth3d"},
})

def get_dataset_and_loader(dset_conf):
    dataset = get_dataset(dset_conf.name)(dset_conf)
    loader = dataset.get_data_loader(dset_conf.get("split", "test"))
    return loader

def rescale(img, scale):
    C, H, W = img.shape[1:]

    # Compute Ratios
    upper_scale = math.ceil(scale)
    ratio = scale / upper_scale

    # Repeat the image to form 2x2 grid (4 images)
    img_tiled = img.repeat(upper_scale * upper_scale, 1, 1, 1)  # Shape: [4, C, H, W]

    # Arrange into 2x2 grid using .view() and .permute()
    img_grid = img_tiled.view(upper_scale, upper_scale, C, H, W).permute(2, 0, 3, 1, 4).reshape(1, C, H * upper_scale, W * upper_scale)

    # Finally rescale properly
    return img_grid[:, :, :int(img_grid.shape[2] * ratio), :int(img_grid.shape[3] * ratio)]

def run_measurement(
    name,
    config_name,
    num_s,
):
    dset_conf = conf[config_name]

    # get model
    lsd_model = get_model("lines.lsd")(conf["lsd_model"])
    faster_lsd_model = get_model("lines.lsd")(conf["fastlsd_model"])

    scales = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

    for scale in scales:
        total_latency_lsd = 0
        total_latecy_faster_lsd = 0

        # get data loader
        dataloader = get_dataset_and_loader(dset_conf)
        for img in tqdm(islice(dataloader, num_s), total=num_s):
            if "image" not in img.keys():
                img["image"] = img["view0"]["image"]

            img["image"] = rescale(img["image"], scale)

            output_lsd = lsd_model(img)
            output_faster_lsd = faster_lsd_model(img)

            total_latency_lsd += output_lsd["latencies"][0]
            total_latecy_faster_lsd += output_faster_lsd["latencies"][0]

        print(f"[{name}] Speedup of faster_lsd over lsd with scale {scale} is : {total_latency_lsd/total_latecy_faster_lsd}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_s", type=int, default=100, help="Number of timing samples."
    )
    args = parser.parse_args()

    run_measurement(
        "Oxford Paris",
        "data_oxford_paris",
        args.num_s,
    )

    run_measurement(
        "HPatches",
        "data_hpatches",
        args.num_s
    )

    run_measurement(
        "RDNIM",
        "data_rdnim",
        args.num_s
    )

    run_measurement(
        "ETH Dataset",
        "data_eth",
        args.num_s,
    )
