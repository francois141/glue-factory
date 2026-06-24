"""
Glue Factory wrapper around the RIPE keypoint detector / descriptor.
https://github.com/fraunhoferhhi/RIPE
"""

from pathlib import Path
import sys

import torch

from ..base_model import BaseModel
from ..utils.misc import pad_and_stack

RIPE_ROOT = Path(__file__).resolve().parents[3] / "external" / "RIPE"
sys.path.append(str(RIPE_ROOT))

from ripe import vgg_hyper

class RIPE(BaseModel):
    default_conf = {
        "model_path": None,
        "threshold": 1.0,
        "top_k": 2048,
        "max_num_keypoints": None,
        "force_num_keypoints": False,
        "dense_outputs": False,
        "descriptor_dim": 256,
        "device": "auto",
    }

    required_data_keys = ["image"]

    def _init(self, conf):
        if conf.dense_outputs:
            raise NotImplementedError("RIPE dense outputs are not supported")

        if conf.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(conf.device)

        model_path = None if conf.model_path is None else Path(conf.model_path)
        self.model = vgg_hyper(model_path=model_path).to(self.device)
        self.model.eval()

        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

        self.set_initialized()

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        elif image.shape[0] > 3:
            image = image[:3]
        return (image.to(self.device) - self.mean) / self.std

    def _forward_single(self, image: torch.Tensor):
        with torch.inference_mode():
            try:
                keypoints, descriptors, scores = self.model.detectAndCompute(
                    self._normalize(image),
                    threshold=self.conf.threshold,
                    top_k=self._num_keypoints,
                )
            except RuntimeError as exc:
                if "No keypoints detected" not in str(exc):
                    raise
                keypoints = torch.empty(0, 2, device=self.device)
                descriptors = torch.empty(
                    0, self.conf.descriptor_dim, device=self.device
                )
                scores = torch.empty(0, device=self.device)

        return keypoints, scores, descriptors

    @property
    def _num_keypoints(self):
        if self.conf.max_num_keypoints is not None:
            return self.conf.max_num_keypoints
        return self.conf.top_k

    def _forward(self, data):
        image = data["image"]
        keypoints = []
        scores = []
        descriptors = []

        for i in range(image.shape[0]):
            img = image[i]

            kpts, sc, desc = self._forward_single(img)
            keypoints.append(kpts)
            scores.append(sc)
            descriptors.append(desc)

        if len(keypoints) == 1:
            keypoints = keypoints[0][None]
            scores = scores[0][None]
            descriptors = descriptors[0][None]
        else:
            keypoints = pad_and_stack(keypoints, pad_dim=-2, mode="zeros")
            scores = pad_and_stack(scores, pad_dim=-1, mode="zeros")
            descriptors = pad_and_stack(descriptors, pad_dim=-2, mode="zeros")

        return {
            "keypoints": keypoints.to(image.device),
            "keypoint_scores": scores.to(image.device),
            "descriptors": descriptors.to(image.device),
        }

    def loss(self, pred, data):
        raise NotImplementedError
