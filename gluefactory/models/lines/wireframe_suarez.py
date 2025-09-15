import argparse
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image

from wireframe_distillation.wireframe_net import WireframeNet

import numpy as np
import torch
import time
from joblib import Parallel, delayed
from pytlsd import lsd, lsd_from_points, lsd_opt
from faster_pytlsd import lsd as fast_lsd
from faster_pytlsd import params_lsd
from gluefactory.utils.image import compute_lsd_image_gradient, extract_non_zero_points_sorted_by_gradient

from ..base_model import BaseModel
from ...settings import DATA_PATH


class WireframeSuarez(BaseModel):
    default_conf = {}

    required_data_keys = ["image"]

    def download_model(self, path):
        import subprocess
        if not path.parent.is_dir():
            path.parent.mkdir(parents=True, exist_ok=True)
        link = "https://raw.githubusercontent.com/iferfra/wireframe-detector/main/checkpoints/checkpoint.pth"
        cmd = ["wget", link, "-O", path]
        print("Downloading ScaleLSD model...")
        subprocess.run(cmd, check=True)

    def _init(self, conf):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_name = "wireframe-suarez"

        ckpt = DATA_PATH / "weights" / self.model_name
        if not ckpt.is_file():
            self.download_model(ckpt)

        config = {"weights": ckpt, "size": [800, 800]}

        self.model = WireframeNet(config)
        self.model.eval().to(device)

    def _forward(self, data):
        results = self.model.forward(data)

        return {
            "lines": torch.tensor(results["line_segments"]).unsqueeze(0),
            "keypoints": results["points"],
            "keypoint_scores": results["scores"],
            "descriptors": results["descriptors"],
        }

    def loss(self):
        raise NotImplementedError

    def is_initialized(self):
        return True