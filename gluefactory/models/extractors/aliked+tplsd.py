"""Wrapper for joint point and line detection with ALIKED and TP-LSD."""

import torch

from gluefactory.models.extractors.aliked import ALIKED
from gluefactory.models.lines.tplsd import TPLSD

from ..base_model import BaseModel


class AlikedTPLSD(BaseModel):
    default_conf = {
        # ALIKED config
        "max_num_keypoints": 2048,
        "detection_threshold": -1,
        "force_num_keypoints": False,
        # TP-LSD config
        "tplsd_variant": "tplite",
        "min_length": 15,
        "max_num_lines": None,
        "force_num_lines": False,
        "tps_thresh": 0.25,
        "tps_lmbd": 0.5,
    }

    def is_initialized(self):
        return True

    def _init(self, conf):
        self.tplsd = TPLSD(
            {
                "tplsd_variant": conf.tplsd_variant,
                "min_length": conf.min_length,
                "max_num_lines": conf.max_num_lines,
                "force_num_lines": conf.force_num_lines,
                "tps_thresh": conf.tps_thresh,
                "tps_lmbd": conf.tps_lmbd,
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
            tplsd_output = self.tplsd(data)
            aliked_output = self.aliked(data)

        return {**tplsd_output, **aliked_output}

    def loss(self, pred, data):
        raise NotImplementedError
