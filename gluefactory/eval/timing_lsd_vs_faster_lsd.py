"""
Performs timing on a model and a dataset. Model and dataset must be specified in a conf file.
Attention: Make sure you configure batch_size and dataset split correctly.

- Ex run: python -m gluefactory.eval.timing_lsd_vs_faster_lsd
"""

import argparse
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from itertools import islice

from gluefactory.datasets import get_dataset
from gluefactory.models import get_model
from gluefactory.utils.tensor import batch_to_device

# TODO: Fix this part and add support for multiple datasets
conf =  OmegaConf.create({
    "lsd_model": { "name": "lines.lsd", "faster_lsd": False},
    "fastlsd_model": {"name": "lines.lsd", "faster_lsd": True},
    "data_rdnim": {"name": "gluefactory.datasets.rdnim"},
    "data_hpatches": {"name": "gluefactory.datasets.hpatches"},
    "data_oxford_paris": {"name": "gluefactory.datasets.oxford_paris_mini_1view_jpldd"},
})

def get_dataset_and_loader(dset_conf):
    dataset = get_dataset(dset_conf.name)(dset_conf)
    loader = dataset.get_data_loader(dset_conf.get("split", "test"))
    return loader

def run_measurement(
    name,
    config_name,
    num_s
):
    dset_conf = conf[config_name]

    # get data loader
    dataloader = get_dataset_and_loader(dset_conf)

    # get model
    lsd_model = get_model("lines.lsd")(conf["lsd_model"])
    faster_lsd_model = get_model("lines.lsd")(conf["fastlsd_model"])

    total_latency_lsd = 0
    total_latecy_faster_lsd = 0

    for img in tqdm(islice(dataloader, num_s), total=num_s):

        if "image" not in img.keys():
            img["image"] = img["view0"]["image"]

        output_lsd = lsd_model(img)
        output_faster_lsd = faster_lsd_model(img)

        total_latency_lsd += output_lsd["latencies"][0]
        total_latecy_faster_lsd += output_faster_lsd["latencies"][0]

    print(f"[{name}] Speedup of faster_lsd over lsd is : {total_latency_lsd/total_latecy_faster_lsd}")

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