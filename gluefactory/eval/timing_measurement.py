"""
Performs timing on a model and a dataset. Model and dataset must be specified in a conf file.
Attention: Make sure you configure batch_size and dataset split correctly.

- Ex run: python -m gluefactory.eval.timing_measurement --conf=gluefactory/configs/timing_eval/jpl.yaml --num_s=100 --device=cuda
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


def get_dataset_and_loader(dset_conf, size=800):
    dset_conf.reshape = size
    dset_conf.square_pad = True
    dataset = get_dataset(dset_conf.name)(dset_conf)
    loader = dataset.get_data_loader(dset_conf.get("split", "test"))
    return loader


def sync_and_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t = time.time()
    return t


def measure_performance(dataloader, model, num_s, device, batch_size):
    # Load a single batch to device
    for img in dataloader:
        input_batched = batch_to_device(img, device, non_blocking=True)
        break

    # Measure latency (single image)
    img_single = {"image": input_batched['image'][:1]}
    start = sync_and_time()
    with torch.no_grad():
        for _ in tqdm(range(num_s)):
            _ = model(img_single)
            if device == "cuda":
                torch.cuda.synchronize()
    end = sync_and_time()
    return (end - start) / num_s * 1e3


def print_performance(latency):
    print(f"Current latency in milliseconds: {latency:.2f}")


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, help="Name of config to run")
    parser.add_argument(
        "--num_s", type=int, default=100, help="Number of timing samples."
    )
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cuda")
    parser.add_argument("--multiple_size", action="store_true")
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

    print_performance(measure_performance(
        dataloader=dataloader,
        model=model,
        num_s=args.num_s,
        device=device,
        batch_size=dataloader.batch_size,
    ))

    if args.multiple_size:
        print("=== Now running the latency over image size ===")

        sizes = [64, 128, 256, 512, 1024, 2048]

        outputs = []
        for size in sizes:
            outputs.append(measure_performance(
                dataloader=get_dataset_and_loader(dset_conf, size=size),
                model=model,
                num_s=args.num_s,
                device=args.device,
                batch_size=dataloader.batch_size,
            ))

        print("=== Sizes ===")
        print(sizes)

        print("=== Latencies ===")
        print(outputs)
