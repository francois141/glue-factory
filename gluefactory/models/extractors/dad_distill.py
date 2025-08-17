"""Gluefactory port of DAD point detector
The code comes from: https://github.com/Parskatt/dad
"""

from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from ..base_model import BaseModel
from ..utils.misc import pad_and_stack

import dad
from dad.loss import MaxDistillLoss
from PIL import Image

class DadDistillDetector(BaseModel):
    default_conf = {
        "max_num_keypoints": 1024,
    }

    def is_initialized(self):
        return True
    
    def _init(self, conf):
        self.dad_light_detector = dad.load_DaDLight()
        self.dad_black_detector = dad.load_DaDDark()
        
        self.max_num_keypoints = conf.max_num_keypoints

        if self.max_num_keypoints == None:
            self.max_num_keypoints = 1024

    def _forward(self, data):
        # See dad.py to implement a similar function
        raise NotImplementedError

    # Returns the KL divergence between the current network we are teaching and the 
    def get_kl_divergence(self, batch, scoremap_student):
        p_teachers = []
        with torch.inference_mode():
            for teacher in [self.dad_black_detector, self.dad_light_detector]:
                scoremap: torch.Tensor = teacher(batch)["scoremap"]
                B, one, H, W = scoremap.shape
                p_teachers.append(
                    scoremap.reshape(B, H * W).softmax(dim=1).reshape(B, 1, H, W)
                )
        p_max = torch.maximum(*p_teachers).clone()
        p_max = p_max / p_max.sum(dim=(-2, -1), keepdim=True)
        scoremap: torch.Tensor = scoremap_student.unsqueeze(1)
        B, one, H, W = scoremap.shape
        log_p_model = scoremap.reshape(B, H * W).log_softmax(dim=1).reshape(B, 1, H, W)
        kl = (
            -(p_max * log_p_model).sum() / B + (p_max * (p_max + 1e-10).log()).sum() / B
        )
        return kl.unsqueeze(0)

    def loss(self, pred, data):
        raise NotImplementedError
