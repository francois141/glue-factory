"""
A generic training script that works with any model and dataset.

Author: Paul-Edouard Sarlin (skydes)
"""

import argparse
import copy
from collections import defaultdict
from pathlib import Path
from pydoc import locate

import numpy as np
from omegaconf import OmegaConf
from gluefactory.visualization.viz2d import plot_images, plot_lines
import cv2

import torch
import matplotlib.pyplot as plt
import numpy as np

from gluefactory.eval.io import load_model

from . import __module_name__, logger
from .datasets import get_dataset

def nicer_display(df):
    return (df) ** (1/4)

def visualize_img_with_gt(name, elem, num=5, offset=0, lim_kpoints=-1):
    plt.figure()
    _, ax = plt.subplots(1, 5, figsize=(16, 8))

    df_pred = elem["predicted_distance_field"]
    df_pred2 = elem["predicted_distance_field2"]
    df = elem["deeplsd_distance_field"]

    ax[0].axis('off')
    ax[0].set_title('Ground Distance Field')
    ax[0].imshow(df)

    ax[1].axis('off')
    ax[1].set_title('Predicted Distance Field')
    ax[1].imshow(df_pred)

    ax[2].axis('off')
    ax[2].set_title('Predicted Distance Field 100k')
    ax[2].imshow(df_pred2)

    image = elem["image"].permute(1,2,0).cpu().numpy()
    image = np.ascontiguousarray((255*image).astype(np.uint8))
    for p in elem["lines"][0]:
        p = p.cpu().numpy().astype(int)
        cv2.line(image, 
                     (p[0,0], p[0,1]), 
                     (p[1,0], p[1,1]), 
                     color=(0, 255, 0), 
                     thickness=2)
    ax[3].axis('off')
    ax[3].set_title('Base lines')
    ax[3].imshow(image)

    image = elem["image"].permute(1,2,0).cpu().numpy()
    image = np.ascontiguousarray((255*image).astype(np.uint8))
    for p in elem["lines2"][0]:
        p = p.cpu().numpy().astype(int)
        cv2.line(image, 
                     (p[0,0], p[0,1]), 
                     (p[1,0], p[1,1]), 
                     color=(0, 255, 0), 
                     thickness=2)
    ax[4].axis('off')
    ax[4].set_title('100k lines')
    ax[4].imshow(image)

    plt.savefig(f"test_{name}.png")
    plt.close()

# Command to launch it
# python -m gluefactory.test_oxparis_100k_output --conf gluefactory/configs/benchmark_jpl_lsd.yaml 
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
    jpldd_model = load_model(conf.model.extractor, "/home/francois/Bureau/glue-factory/outputs/training/TEST_TRAIN").to("cpu")
    jpldd_model.eval()

    jpldd_model_2 = load_model(conf.model.extractor, "/home/francois/Bureau/glue-factory/outputs/training/TRAIN_100k_DISTRIBUTED_4").to("cpu")
    jpldd_model_2.eval()
    print("Model loaded")

    for it, data in enumerate(train_loader):
        print(f"=== Running iteration: {it} ===")
        dataset = {}

        output_jpldd = jpldd_model({"image": data["image"].to("cpu").unsqueeze(0)})
        output_jpldd2 = jpldd_model_2({"image": data["image"].to("cpu").unsqueeze(0)})

        # Load image
        dataset["image"] = data["image"]
        dataset["lines"] = output_jpldd["lines"]
        dataset["lines2"] = output_jpldd2["lines"]

        #  Predicted distance field
        img_np = output_jpldd["line_distancefield"].detach().numpy()[0]
        dataset["predicted_distance_field"] = img_np

        img_np = output_jpldd2["line_distancefield"].detach().numpy()[0]
        dataset["predicted_distance_field2"] = img_np

        img_np = data["deeplsd_distance_field"].numpy()

        dataset["deeplsd_distance_field"] = img_np

        visualize_img_with_gt(it, dataset)
