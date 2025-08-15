"""Gluefactory port of DAD point detector
The code comes from: https://github.com/Parskatt/dad
"""

from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from ..base_model import BaseModel
from ..utils.misc import pad_and_stack

import dad
from PIL import Image

class SuperPoint(BaseModel):
    default_conf = {}

    def is_initialized(self):
        return True

    def _init(self, conf):
        self.dad_detector = dad.load_DaD()

    def _forward(self, data):
        with torch.no_grad():
            detections = self.dad_detector.detect(
                {"image": data["image"]}, 
                num_keypoints=512,
                return_dense_probs=True
            )

            _, _, image_height, image_width = data["image"].shape

            keypoints = detections["keypoints"]

            x_px = ((keypoints[..., 0] + 1) / 2) * image_width
            y_px = ((keypoints[..., 1] + 1) / 2) * image_height
            keypoints_px = torch.stack((x_px, y_px), dim=-1)

            return {
                "keypoints": keypoints_px,
                "keypoint_scores": detections["keypoint_probs"],
                "heatmap": detections["dense_probs"],
            }


    def loss(self, pred, data):
        raise NotImplementedError
