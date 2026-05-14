# Integration of PlNet joint points and lines detector: https://github.com/sair-lab/PLNet
# Adapted to work with arbitrary resolution input


import sys
from pathlib import Path

import torch
from ...settings import DATA_PATH
from ..base_model import BaseModel

# Add PLNet to sys.path
plnet_path = Path(__file__).parent.parent.parent.parent / "other/PLNet"
sys.path.append(str(plnet_path))

try:
    from hawp.fsl.config import cfg as model_config
    from hawp.fsl.model.build import build_model
    from hawp.fsl.dataset.build import Normalize
    print("Successfully imported PLNet components from ", plnet_path)
except ImportError as e:
    print(f"Warning: Could not import PLNet from {plnet_path}. Make sure the submodule is initialized and dependencies (yacs, easydict) are installed.")
    print(f"Error: {e}")


class PLNet(BaseModel):
    default_conf = {
        "checkpoint_url": "https://entuedu-my.sharepoint.com/:u:/g/personal/kuan_xu_staff_main_ntu_edu_sg/EbQy7pSPVNFDrP81aloP-O8BA3W0HlOqFsTi6p20KGH9xA?e=mFgVdU&download=1",
        "config_path": "other/PLNet/configs/plnet.yaml",
        "max_num_junctions": 512,
        "max_num_lines": 512,
        "num_keypoints": None,  # If set, return exactly this many keypoints (top by score, no threshold)
    }

    def is_initialized(self):
        return True

    def _init(self, conf):
        """
        Initialize PlNet model - download model weights if not accessible yet and load model.
        """
        print("Initialize PLNet....")
        root = Path(__file__).parent.parent.parent.parent
        ckpt_path = DATA_PATH / "weights/plnet.pth"
        if not ckpt_path.is_file():
            self.download_model(ckpt_path)

        # Load config
        full_config_path = root / conf.config_path
        model_config.merge_from_file(str(full_config_path))
        self.model_conf = model_config

        # Initialize model
        self.net = build_model(model_config).eval()
        self.image_normalizer = Normalize(self.model_conf.DATASETS.IMAGE.PIXEL_MEAN, self.model_conf.DATASETS.IMAGE.PIXEL_STD,
                                           self.model_conf.DATASETS.IMAGE.TO_255)

        # Load weights
        state_dict = torch.load(ckpt_path, map_location="cpu")
        self.net.load_state_dict(state_dict["model"])

    def download_model(self, path):
        import subprocess

        if not path.parent.is_dir():
            path.parent.mkdir(parents=True, exist_ok=True)
        link = self.conf.checkpoint_url
        cmd = ["wget", link, "-O", str(path)]
        print(f"Downloading PLNet model from {link}...")
        subprocess.run(cmd, check=True)

    def _forward(self, data):
        """
        Forward pass of PlNet. Wraps PlNet forward and processes lines and points that are output to glue-factory
        compatible format.

        Points detection: internally based on heatmap after nms gets 1000 best points by scores and only keeps if confidence > 0.1 >> Cannot control number of points directly.
        Only max number of points and lines possible.
        """
        image = data["image"]
        
        batch_size, C, H, W = image.shape
        assert C == 3, "PLNet only supports 3-channel RGB images"
        assert batch_size == 1, "PLNet forward_test only supports batch_size=1 currently"

        # special PLNet normalization
        # 1) normalize
        if image.max() > 1.0:
            image /= 255.0
        # 2) mean-std normalize for 3 channels
        if not self.model_conf.MODEL.NAME == "PointLine":
            print("Warning: PLNet model name is not PointLine, normalize....")
            image = self.image_normalizer(image)

        # Pad to multiple of 64 to avoid RuntimeError in UNet
        div = 64
        pad_h = (div - H % div) % div
        pad_w = (div - W % div) % div
        if pad_h > 0 or pad_w > 0:
            # Pad with zeros (bottom and right)
            image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h))
        
        H_padded, W_padded = image.shape[2:]

        # Meta information required by PLNet
        meta = {
            'width': W_padded,
            'height': H_padded,
            'filename': ''
            }

        with torch.no_grad():
            out, _ = self.net.forward_test(image, [meta], num_keypoints=self.conf.num_keypoints)
                
        juncs = out['juncs_pred'] # [N, 2]
        junc_scores = out['juncs_score'] # [N]
        juncs_desc = out.get('juncs_desc')  # [N, 256] or None

        # Filter junctions by padding and threshold
        if pad_h > 0 or pad_w > 0:
            mask = (juncs[:, 0] < W) & (juncs[:, 1] < H)
            juncs = juncs[mask]
            junc_scores = junc_scores[mask]
            if juncs_desc is not None:
                juncs_desc = juncs_desc[mask]

        # Keep top-k junctions (skip if num_keypoints is explicitly set)
        if self.conf.num_keypoints is None and len(juncs) > self.conf.max_num_junctions:
            junc_scores, indices = torch.topk(junc_scores, self.conf.max_num_junctions)
            juncs = juncs[indices]
            if juncs_desc is not None:
                juncs_desc = juncs_desc[indices]

        # Lines
        lines = out['lines_pred'].reshape(-1, 2, 2) # [M, 2, 2]
        line_scores = out['lines_score'] # [M]

        # Filter lines by padding and threshold
        if pad_h > 0 or pad_w > 0:
            mask = (lines[:, 0, 0] < W) & (lines[:, 0, 1] < H) & \
                    (lines[:, 1, 0] < W) & (lines[:, 1, 1] < H)
            lines = lines[mask]
            line_scores = line_scores[mask]

        # Keep top-k lines
        if len(lines) > self.conf.max_num_lines:
            line_scores, indices = torch.topk(line_scores, self.conf.max_num_lines)
            lines = lines[indices]

        result = {
            "keypoints": juncs[None],
            "keypoint_scores": junc_scores[None],
            "lines": lines[None],
            "line_scores": line_scores[None],
            "valid_lines": torch.ones_like(line_scores)[None],
        }

        if juncs_desc is not None:
            result["descriptors"] = juncs_desc[None]  # [1, N, 256]

        return result

    def loss(self, pred, data):
        """
        Loss not needed for pure inference integration.
        """
        raise NotImplementedError