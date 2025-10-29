"""
Performs timing on a model and a dataset. Model and dataset must be specified in a conf file.
Attention: Make sure you configure batch_size and dataset split correctly.

- Ex run: python -m gluefactory.eval.timing_measurement --conf=gluefactory/configs/timing_conf.yaml --num_s=100 --device=cuda
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from tqdm import tqdm

from gluefactory.datasets import get_dataset
from gluefactory.models import get_model
from gluefactory.utils.tensor import batch_to_device

default_conf = {"model": {"name": ""}, "dataset": {"name": ""}}


def get_dataset_and_loader(dset_conf):
    dataset = get_dataset(dset_conf.name)(dset_conf)
    loader = dataset.get_data_loader(dset_conf.get("split", "test"))
    return loader


def sync_and_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t = time.time()
    return t


def run_measurement(
    dataloader, model, num_s, name, device, batch_size
):
    print(f"Batch_size: {batch_size}")

    count = 0
    timings = []
    input_batched = 0
    for img in tqdm(
        dataloader,
        total=1
    ):
        input_batched = batch_to_device(img, device, non_blocking=True)
        break


    start = sync_and_time()

    with torch.no_grad():
        for _ in tqdm(range(num_s)):
            _ = model(input_batched)
            if device == "cuda":
                torch.cuda.synchronize()

    end = sync_and_time()
    print("Current throughput in detections/s: " + str(num_s * batch_size / (end - start)))

    img_single =  {"image": input_batched['image'][:1]}

    start = sync_and_time()

    with torch.no_grad():
        for _ in tqdm(range(num_s)):
            _ = model(img_single)
            if device == "cuda":
                torch.cuda.synchronize()

    end = sync_and_time()
    print("Current latency in milliseconds: " + str((end - start) / num_s * 1e3))

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, help="Name of config to run")
    parser.add_argument(
        "--num_s", type=int, default=100, help="Number of timing samples."
    )
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cuda")
    args = parser.parse_args()

    default_conf = OmegaConf.create(default_conf)
    conf = OmegaConf.merge(default_conf, OmegaConf.load(args.conf))
    dset_conf = conf["data"]
    model_conf = conf["model"]
    model_is_jpl = "joint_point_line_extractor" in model_conf["name"]

    print("NUMBER OF SAMPLES: ", args.num_s)
    print("CONF TO TEST: ", args.conf)
    print("Dataset: ", dset_conf.name)
    print("--Split: ", dset_conf.split)
    print("Model: ", model_conf.name)


    # get data loader
    dataloader = get_dataset_and_loader(dset_conf)

    if not torch.cuda.is_available():
        args.device = "cpu"

    if args.device == "cuda":
        assert torch.cuda.is_available()
    elif args.device == "mps":
        assert torch.backends.mps.is_built()

    device = args.device
    print(f"Using Device: {device}")

    # get model
    model = get_model(model_conf.name)(model_conf)
    model.eval()
    model.to(device)

    if model_is_jpl:
        print(f"Model contains : {model.get_numer_of_parameters():.3e} parameters overall")
    else:
        print(f"Model contains : {count_parameters(model):.3e} parameters overall")

    run_measurement(
        dataloader=dataloader,
        model=model,
        num_s=args.num_s,
        name=model_conf.name,
        device=device,
        batch_size=dataloader.batch_size,
    )
