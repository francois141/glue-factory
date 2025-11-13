"""
Wrapper for joint point and line detection with SuperPoint and ScaleLSD.
"""

import torch

from gluefactory.models.extractors.superpoint import SuperPoint
from gluefactory.models.lines.scalelsd import ScaleLSD

from ..base_model import BaseModel


class SuperpointScaleLSD(BaseModel):
    default_conf = {
        "max_num_keypoints": 2048
    }

    def is_initialized(self):
        return True

    def _init(self, conf):
        self.scalelsd = ScaleLSD({}).eval()
        self.superpoint = SuperPoint(conf).eval()

    def _forward(self, data):
        with torch.no_grad():
            scalelsd_output = self.scalelsd(data)
            superpoint_output = self.superpoint(data)

        return {**scalelsd_output, **superpoint_output}

    def loss(self, pred, data):
        raise NotImplementedError
