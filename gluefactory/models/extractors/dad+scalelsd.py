"""
Wrapper for joint point and line detection with dad and scalelsd.
"""

import torch

from gluefactory.models.extractors.dad import DadDetector
from gluefactory.models.lines.scalelsd import ScaleLSD

from ..base_model import BaseModel


class DaDScaleLSD(BaseModel):
    default_conf = {
        "max_num_keypoints": 1024,
        "max_num_lines": None,
        "force_num_lines": False,
    }

    def is_initialized(self):
        return True

    def _init(self, conf):
        self.dad_detector = DadDetector({
            "max_num_keypoints": conf.max_num_keypoints
        }).eval()
        line_conf = {}
        if conf.max_num_lines is not None:
            line_conf["max_num_lines"] = conf.max_num_lines
            line_conf["force_num_lines"] = conf.force_num_lines
        self.scalelsd_detector = ScaleLSD(line_conf).eval()

    def _forward(self, data):
        with torch.no_grad():
            dad_output = self.dad_detector(data)
            scalelsd_output = self.scalelsd_detector(data)

        return {**dad_output, **scalelsd_output}

    def loss(self, pred, data):
        raise NotImplementedError
