"""Wrapper for joint point and line detection with ALIKED and M-LSD."""

import torch

from gluefactory.models.extractors.aliked import ALIKED
from gluefactory.models.lines.mlsd import MLSD

from ..base_model import BaseModel


class AlikedMLSD(BaseModel):
    default_conf = {
        # ALIKED config
        "max_num_keypoints": 2048,
        "detection_threshold": -1,
        "force_num_keypoints": False,
        # M-LSD config
        "mlsd_size": "tiny",
        "input_size": 512,
        "score_thr": 0.1,
        "dist_thr": 20.0,
        "min_length": 15,
        "max_num_lines": None,
        "force_num_lines": False,
    }

    def is_initialized(self):
        return True

    def _init(self, conf):
        self.mlsd = MLSD(
            {
                "mlsd_size": conf.mlsd_size,
                "input_size": conf.input_size,
                "score_thr": conf.score_thr,
                "dist_thr": conf.dist_thr,
                "min_length": conf.min_length,
                "max_num_lines": conf.max_num_lines,
                "force_num_lines": conf.force_num_lines,
            }
        ).eval()
        self.aliked = ALIKED(
            {
                "max_num_keypoints": conf.max_num_keypoints,
                "detection_threshold": conf.detection_threshold,
                "force_num_keypoints": conf.force_num_keypoints,
            }
        ).eval()

    def _forward(self, data):
        with torch.no_grad():
            mlsd_output = self.mlsd(data)
            aliked_output = self.aliked(data)

        return {**mlsd_output, **aliked_output}

    def loss(self, pred, data):
        raise NotImplementedError
