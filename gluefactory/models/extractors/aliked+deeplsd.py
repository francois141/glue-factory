"""Gluefactory port of DAD point detector
The code comes from: https://github.com/Parskatt/dad
"""

from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import torch

from gluefactory.models.lines.deeplsd import DeepLSD
from gluefactory.models.extractors.aliked import ALIKED

from ..base_model import BaseModel

class AlikedDeepLSD(BaseModel):
    default_conf = {}

    def is_initialized(self):
        return True

    def _init(self, conf):
        self.deeplsd = DeepLSD({}).eval()
        self.aliked = ALIKED({
            "max_num_keypoints": 1024,
            "detection_threshold": 0, 
            "force_num_keypoints": True
        }).eval()

    def _forward(self, data):
        with torch.no_grad():
            deeplsd_output = self.deeplsd(data)
            aliked_output = self.aliked(data)

        return {**deeplsd_output, **aliked_output}

    def loss(self, pred, data):
        raise NotImplementedError
