"""Gluefactory port of DAD point detector
The code comes from: https://github.com/Parskatt/dad
"""

from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import dad
import torch
import torch.nn as nn
from DeDoDe import dedode_descriptor_B, dedode_descriptor_G, dedode_detector_L
from DeDoDe.matchers.dual_softmax_matcher import DualSoftMaxMatcher
from PIL import Image

from ..base_model import BaseModel
from ..utils.misc import pad_and_stack


class DeDoDeDetector(BaseModel):
    default_conf = {
        "max_num_keypoints": 1024,
    }

    def is_initialized(self):
        return True

    def _init(self, conf):

        self.dedode_detector = dedode_detector_L(weights=None)

        self.descriptor_lightweight = dedode_descriptor_B(weights=None)
        # self.descriptor_heavyweight = dedode_descriptor_G(weights = None)

        self.max_num_keypoints = 1024

    def _forward(self, data):
        with torch.no_grad():
            input_data = {"image": data["image"]}

            detections = self.dedode_detector.detect(
                input_data, num_keypoints=self.max_num_keypoints
            )

            mask = self.dedode_detector.detect_dense(input_data)[
                "dense_keypoint_logits"
            ]

            descriptors = self.descriptor_lightweight.describe_keypoints(
                input_data, detections["keypoints"]
            )["descriptions"]

            _, _, image_height, image_width = data["image"].shape
            keypoints = detections["keypoints"]

            x_px = ((keypoints[..., 0] + 1) / 2) * image_width
            y_px = ((keypoints[..., 1] + 1) / 2) * image_height
            keypoints_px = torch.stack((x_px, y_px), dim=-1)

            return {
                "keypoints": keypoints_px,
                "keypoint_scores": detections["confidence"],
                "heatmap": mask,
                "descriptors": descriptors,
            }

    def describe_keypoints(self, input_data, keypoints):
        return self.descriptor_lightweight.describe_keypoints(input_data, keypoints)[
            "descriptions"
        ]

    def loss(self, pred, data):
        raise NotImplementedError
