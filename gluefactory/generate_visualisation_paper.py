"""
Minimal script to load a model and run it on a single image.

Author: Adapted from Paul-Edouard Sarlin (skydes)
"""

import argparse
from pathlib import Path

import cv2
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from gluefactory.visualization.viz2d import plot_images, plot_lines
from gluefactory.eval.io import load_model, get_model


def visualize_img_with_gt(output, input_image, save_path):
    plt.figure()
    _, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Right: detected lines
    plot_images([input_image], [], cmaps="gray")
    plot_lines([output["lines"][0]], indices=range(1))

    plt.savefig(save_path)
    plt.close()


def get_jpl_model():
    return load_model(
        {
            'name': 'extractors.joint_point_line_extractor',
            'line_df_decoder_channels': 64,
            'training': {'do': False},
            'max_num_keypoints': 2048,
            'line_detection': {'do': True, 'conf': {'lsd_type': 'old', 'sigma': 0.9}},
            'checkpoint': 'assets/jpl_best.tar'
        },
        "./assets/jpl_best_with_points.tar"
    ).to("cpu")

def get_scalelsd_model():
    return get_model('lines.scalelsd')({
        'name': 'lines.scalelsd',
        'threshold': 10
    }).to("cpu")

def get_wireframe_model():
    return get_model('lines.wireframe_suarez')({
        'name': 'lines.wireframe_suarez'
    }).to("cpu")

def get_deeplsd_model():
    return get_model('lines.deeplsd')({
        'name': 'lines.deeplsd',
       ' model_conf':
            {'detect_lines': True,
                    'line_detection_params': {
                        'use_img_grad_angle': False,
                        'merge': False,
                        'grad_nfa': True,
                        'filtering': True,
                        'faster_lsd': False,
                    }
            }
    }).to("cpu")

def get_lsd_model():
    return get_model('lines.lsd')({
        'name': 'lines.lsd'
    }).to("cpu")

# Command to run it: python -m gluefactory.generate_visualisation_paper
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="assets/boat1.png", help="Path to input image")
    args = parser.parse_args()


    # Load model
    print("Loading model...")
    model = get_lsd_model()
    model.eval()
    print("Model loaded.")

    # Load image in RGB (fix: convert from BGR)
    img_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Convert to tensor [C, H, W], normalized to [0,1]
    img_tensor = torch.from_numpy(img_rgb).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

    # Visualize results
    models = [
        (get_lsd_model, "lsd"),
        (get_deeplsd_model, "deeplsd"),
        (get_jpl_model, "jpl"),
        (get_scalelsd_model, "scalelsd"),
        (get_wireframe_model, "wireframe"),
    ]

    for getter, name in models:
        model = getter()
        # Run inference
        with torch.no_grad():
            output = model({"image": img_tensor})

        path = f"generated_jpl_images/example_{name}.pdf"
        print("Visualization saved to {}".format(path))
        visualize_img_with_gt(output, img_rgb, path)


