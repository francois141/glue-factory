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

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation
import flow_vis

from gluefactory.eval.io import get_eval_parser, load_model, parse_eval_args

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
    return (df) ** (1/4)

def get_flow_vis(df, ang, line_neighborhood=5):
    norm = line_neighborhood + 1 - np.clip(df, 0, line_neighborhood)
    flow_uv = np.stack([norm * np.cos(ang), norm * np.sin(ang)], axis=-1)
    flow_img = flow_vis.flow_to_color(flow_uv, convert_to_bgr=False)
    return flow_img

def visualize_img_with_gt(name, dset, num=5, offset=0, lim_kpoints=-1):
    plt.figure()
    idxs = list(range(offset, offset+num, 1))
    _, ax = plt.subplots(1, 3, figsize=(20, 8))
    for i in idxs:
        if i >= 1:
            break
        elem = dset[i]

        df_pred = elem["predicted_distance_field"]
        df = elem["deeplsd_distance_field"]

        ax[0].axis('off')
        ax[0].set_title('Ground Distance Field')
        ax[0].imshow(df)

        ax[1].axis('off')
        ax[1].set_title('Predicted Distance Field')
        ax[1].imshow(df_pred)

        ax[2].axis('off')
        ax[2].set_title('Original')
        ax[2].imshow(elem["image"].permute(1,2,0))

    plt.savefig(f"test_{name}.png")
    plt.show()
    plt.close()

# Command to launch it
# python -m gluefactory.test_oxparis_100k_output --conf gluefactory/configs/train_jpl_oxparis_100k.yaml 
# python -m gluefactory.test_oxparis_100k_output --conf gluefactory/configs/train_jpl_oxparis_base.yaml 
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

    print("Loading train dataset")
    train_loader = dataset.get_dataset("train")
    print("Train dataset loaded")

    print("Loading model")
    jpldd_model = load_model(conf.model, "/cluster/scratch/fcosta/outputs/training/TEST_TRAIN/checkpoint_best.tar").to("cpu")
    jpldd_model.eval()
    print("Model loaded")

    for it, data in enumerate(train_loader):
        print(f"=== Running iteration: {it} ===")
        dataset = {}

        output_jpldd = jpldd_model({"image": data["image"].to("cpu").unsqueeze(0)})
    
        # Load image
        dataset["image"] = data["image"]

        #  Predicted distance field
        img_np = output_jpldd["line_distancefield"].detach().numpy()[0]
        img_np = nicer_display(img_np) * 255
        dataset["predicted_distance_field"] = -img_np

        img_np = data["deeplsd_distance_field"].numpy()
        img_np = nicer_display(img_np) * 255
        dataset["deeplsd_distance_field"] = -img_np

        visualize_img_with_gt(it, [dataset])
