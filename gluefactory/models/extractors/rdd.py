"""
Glue Factory wrapper around the bundled RDD keypoint detector / descriptor.
"""

from pathlib import Path
import sys

import numpy as np
import torch

from ..base_model import BaseModel
from ..utils.misc import pad_and_stack

_ROOT = Path(__file__).resolve().parents[3]
_RDD_ROOT = _ROOT / "external" / "rdd"
if str(_RDD_ROOT) not in sys.path:
    sys.path.append(str(_RDD_ROOT))

try:
    from RDD.RDD import build as build_rdd
    from RDD.utils import read_config
except Exception as exc:  # pragma: no cover - surfaced in _init
    build_rdd = None
    read_config = None
    _RDD_IMPORT_ERROR = exc
else:
    _RDD_IMPORT_ERROR = None


class RDD(BaseModel):
    default_conf = {
        "config_path": str(_RDD_ROOT / "configs" / "default.yaml"),
        "weights": None,
        "detection_threshold": 0.1,
        "max_num_keypoints": 4096,
        "force_num_keypoints": False,
        "device": "auto",
    }

    required_data_keys = ["image"]

    def _init(self, conf):
        if build_rdd is None or read_config is None:
            raise ImportError(
                "Could not import RDD from external/amogh_rdd. "
                "Make sure the repo was cloned into external/amogh_rdd."
            ) from _RDD_IMPORT_ERROR

        config = read_config(conf.config_path)
        if conf.device == "auto":
            config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            config["device"] = conf.device

        config["detection_threshold"] = conf.detection_threshold
        config["top_k"] = conf.max_num_keypoints

        weights = conf.weights
        if weights is not None:
            weights = str(weights)
            if not Path(weights).is_file():
                raise FileNotFoundError(f"RDD weights not found: {weights}")

        self.model = build_rdd(config, weights)
        self.set_initialized()

    def _to_model_input(self, image: torch.Tensor) -> torch.Tensor:
        image = self._to_uint8_image(image)
        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
        return image.unsqueeze(0)

    def _to_uint8_image(self, image: torch.Tensor) -> np.ndarray:
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        elif image.shape[0] > 3:
            image = image[:3]

        image = image.detach().cpu().permute(1, 2, 0).contiguous().numpy()
        if image.max() <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image[..., ::-1].copy()

    def _forward_single(self, image: torch.Tensor):
        out = self.model.extract(self._to_model_input(image))[0]
        keypoints = out["keypoints"]
        scores = out["scores"]
        descriptors = out["descriptors"]

        if (
            self.conf.max_num_keypoints is not None
            and len(keypoints) > self.conf.max_num_keypoints
        ):
            keep = torch.topk(scores, self.conf.max_num_keypoints).indices
            keypoints = keypoints[keep]
            scores = scores[keep]
            descriptors = descriptors[keep]

        return keypoints, scores, descriptors

    def _forward(self, data):
        image = data["image"]
        keypoints = []
        scores = []
        descriptors = []

        for i in range(image.shape[0]):
            img = image[i]
            if "image_size" in data:
                w, h = data["image_size"][i]
                img = img[:, : int(h), : int(w)]
            kpts, sc, desc = self._forward_single(img)
            keypoints.append(kpts)
            scores.append(sc)
            descriptors.append(desc)

        if self.conf.force_num_keypoints and self.conf.max_num_keypoints is not None:
            target_length = self.conf.max_num_keypoints
            for i in range(len(keypoints)):
                current = len(keypoints[i])
                if current < target_length:
                    pad_n = target_length - current
                    keypoints[i] = torch.cat(
                        [
                            keypoints[i],
                            torch.zeros(pad_n, 2, device=keypoints[i].device),
                        ],
                        dim=0,
                    )
                    scores[i] = torch.cat(
                        [scores[i], torch.zeros(pad_n, device=scores[i].device)], dim=0
                    )
                    descriptors[i] = torch.cat(
                        [
                            descriptors[i],
                            torch.zeros(
                                pad_n,
                                descriptors[i].shape[-1],
                                device=descriptors[i].device,
                            ),
                        ],
                        dim=0,
                    )
                elif current > target_length:
                    keep = torch.topk(scores[i], target_length).indices
                    keypoints[i] = keypoints[i][keep]
                    scores[i] = scores[i][keep]
                    descriptors[i] = descriptors[i][keep]

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
