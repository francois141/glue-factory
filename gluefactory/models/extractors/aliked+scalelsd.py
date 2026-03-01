"""
Wrapper for joint point and line detection with ALIKED and ScaleLSD.
"""

import torch

from gluefactory.models.extractors.aliked import ALIKED
from gluefactory.models.lines.scalelsd import ScaleLSD

from ..base_model import BaseModel


class AlikedScaleLSD(BaseModel):
    default_conf = {
        "max_num_keypoints": 2048,
        "detection_threshold": -1,
        "force_num_keypoints": False,
        "max_num_lines": None,
        "force_num_lines": False,
    }

    def is_initialized(self):
        return True

    def _init(self, conf):
        line_conf = {}
        if conf.max_num_lines is not None:
            line_conf["max_num_lines"] = conf.max_num_lines
            line_conf["force_num_lines"] = conf.force_num_lines
        self.scalelsd = ScaleLSD(line_conf).eval()
        self.aliked = ALIKED(
            {
                "max_num_keypoints": conf.max_num_keypoints,
                "detection_threshold": conf.detection_threshold,
                "force_num_keypoints": conf.force_num_keypoints,
            }
        ).eval()

    def _forward(self, data):
        with torch.no_grad():
            scalelsd_output = self.scalelsd(data)
            aliked_output = self.aliked(data)

        return {**scalelsd_output, **aliked_output}

    def loss(self, pred, data):
        raise NotImplementedError
