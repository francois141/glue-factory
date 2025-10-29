"""Gluefactory port of DAD point detector
The code comes from: https://github.com/Parskatt/dad
"""

import dad
import torch
from DeDoDe import dedode_descriptor_B

from ..base_model import BaseModel


class DadDetector(BaseModel):
    default_conf = {
        "max_num_keypoints": 1024,
    }

    def is_initialized(self):
        return True

    def _init(self, conf):
        self.dad_detector = dad.load_DaD()
        self.max_num_keypoints = conf.max_num_keypoints

        self.descriptor_lightweight = dedode_descriptor_B(weights=None)

        if self.max_num_keypoints == None:
            self.max_num_keypoints = 1024

    def _forward(self, data):
        with torch.no_grad():
            input_data = {"image": data["image"]}

            detections = self.dad_detector.detect(
                input_data,
                num_keypoints=self.max_num_keypoints,
                return_dense_probs=True,
            )

            _, _, image_height, image_width = data["image"].shape

            keypoints = detections["keypoints"]

            x_px = ((keypoints[..., 0] + 1) / 2) * image_width
            y_px = ((keypoints[..., 1] + 1) / 2) * image_height
            keypoints_px = torch.stack((x_px, y_px), dim=-1)

            descriptors = self.descriptor_lightweight.describe_keypoints(
                input_data, detections["keypoints"]
            )["descriptions"]

            return {
                "keypoints": keypoints_px,
                "keypoint_scores": detections["keypoint_probs"],
                "heatmap": detections["dense_probs"],
                "descriptors": descriptors,
            }

    def loss(self, pred, data):
        raise NotImplementedError
