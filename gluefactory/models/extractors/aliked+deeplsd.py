"""Gluefactory port of DAD point detector
The code comes from: https://github.com/Parskatt/dad
"""

from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import torch

from gluefactory.models.extractors.aliked import ALIKED
from gluefactory.models.lines.deeplsd import DeepLSD

from ..base_model import BaseModel


class AlikedDeepLSD(BaseModel):
    default_conf = {
        "max_num_keypoints": 1024,
        "detection_threshold": 0,
        "force_num_keypoints": True,
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
        self.deeplsd = DeepLSD(line_conf).eval()
        self.aliked = ALIKED(
            {
                "max_num_keypoints": conf.max_num_keypoints,
                "detection_threshold": conf.detection_threshold,
                "force_num_keypoints": conf.force_num_keypoints,
            }
        ).eval()

    def _forward(self, data):
        with torch.no_grad():
            deeplsd_output = self.deeplsd(data)
            aliked_output = self.aliked(data)

        return {**deeplsd_output, **aliked_output}

    def loss(self, pred, data):
        raise NotImplementedError
