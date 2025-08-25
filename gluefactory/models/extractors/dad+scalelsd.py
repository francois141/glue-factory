"""Gluefactory port of DAD point detector
The code comes from: https://github.com/Parskatt/dad
"""

from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import torch

from gluefactory.models.lines.scalelsd import ScaleLSD
from gluefactory.models.extractors.dad import DadDetector

from ..base_model import BaseModel

class DaDScaleLSD(BaseModel):
    default_conf = {}

    def is_initialized(self):
        return True

    def _init(self, conf):
        self.dad_detector = DadDetector({}).eval()
        self.scalelsd_detector = ScaleLSD({}).eval()

    def _forward(self, data):
        with torch.no_grad():
            dad_output = self.dad_detector(data)
            scalelsd_output = self.scalelsd_detector(data)

        return {**dad_output, **scalelsd_output}

    def loss(self, pred, data):
        raise NotImplementedError
