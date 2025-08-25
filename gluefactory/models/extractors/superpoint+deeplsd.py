"""Gluefactory port of DAD point detector
The code comes from: https://github.com/Parskatt/dad
"""

from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import torch

from gluefactory.models.lines.deeplsd import DeepLSD
from gluefactory.models.extractors.superpoint import SuperPoint

from ..base_model import BaseModel

class SuperpointDeepLSD(BaseModel):
    default_conf = {}

    def is_initialized(self):
        return True

    def _init(self, conf):
        self.deeplsd = DeepLSD({}).eval()
        self.superpoint = SuperPoint({"max_num_keypoints": 1024}).eval()

    def _forward(self, data):
        with torch.no_grad():
            deeplsd_output = self.deeplsd(data)
            superpoint_output = self.superpoint(data)

        return {**deeplsd_output, **superpoint_output}

    def loss(self, pred, data):
        raise NotImplementedError
