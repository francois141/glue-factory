"""
Gluefactory implementation of joint point detector using deeplsd and superpoint
"""

from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import torch

from gluefactory.models.extractors.superpoint import SuperPoint
from gluefactory.models.lines.deeplsd import DeepLSD

from ..base_model import BaseModel


class SuperpointDeepLSD(BaseModel):
    default_conf = {
        "max_num_keypoints": 1024,
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
        self.superpoint = SuperPoint(
            {"max_num_keypoints": conf.max_num_keypoints}
        ).eval()

    def _forward(self, data):
        with torch.no_grad():
            deeplsd_output = self.deeplsd(data)
            superpoint_output = self.superpoint(data)

        return {**deeplsd_output, **superpoint_output}

    def loss(self, pred, data):
        raise NotImplementedError
