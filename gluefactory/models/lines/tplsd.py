"""
TP-LSD wrapper. Uses the TP-LSD submodule at other/TPLSD (auto-initialized).
See: https://github.com/Siyuada7/TP-LSD

TP-LSD (Tri-Points Based Line Segment Detector) is a deep learning-based line detector
that uses a tri-points representation (root-point and two endpoints) for line detection.

Output format:
    - lines: [B, N, 2, 2] tensor (or list) with line endpoints in (x, y) pixel coordinates
    - line_scores: [B, N] tensor (or list) with scores (sqrt of line length)
    - valid_lines: [B, N] bool tensor (or list) indicating valid vs padded lines
"""

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from ..base_model import BaseModel

# Repository root (glue-factory-new/)
_REPO_ROOT = Path(__file__).resolve().parents[3]
# Default location for TP-LSD submodule
_DEFAULT_TPLSD_ROOT = _REPO_ROOT / "other" / "TPLSD"
# DCNv2 extension (separate submodule at other/dcnv2)
_DCNV2_PATH = _REPO_ROOT / "other" / "dcnv2"


class TPLSD(BaseModel):
    """
    TP-LSD (Tri-Points Based Line Segment Detector) line detector.
    
    This is a wrapper around the TP-LSD implementation from:
    https://github.com/Siyuada7/TP-LSD
    
    Setup requirements:
        1. Initialize submodules: git submodule update --init other/TPLSD other/dcnv2
           (done automatically if the directories are empty)
        2. Build DCNv2: cd other/dcnv2 && python setup.py build_ext --inplace
        3. Download pretrained weights (Res160.pth, Res320.pth, or Res512.pth)
           and place them in other/TPLSD/pretraineds/
    
    Configuration:
        - tplsd_variant: Model variant ("tp320", "tplite", or "tp512")
            * "tp320": Res320 model, 320x320 input, uses Res320.pth
            * "tplite": Res160 model (lite version), 320x320 input, uses Res160.pth
            * "tp512": Res320 model, 512x512 input, uses Res512.pth
            Note: Hourglass model ("hg") is not currently supported
        - tplsd_root: Path to TP-LSD repo (default: other/TPLSD submodule)
        - min_length: Minimum line length in pixels (default: 15)
        - max_num_lines: Maximum number of lines to return (default: None = no limit)
        - force_num_lines: If True, pad output to max_num_lines (default: False)
        - tps_thresh: Threshold for tri-points detection (default: 0.25, matches official demo)
        - tps_lmbd: Lambda parameter for line segmentation (default: 0.5, matches official demo)
    """
    default_conf = {
        "tplsd_variant": "tplite",  # "tp320" | "tplite" | "tp512"
        "tplsd_root": None,  # Path to TP-LSD repo; if None, uses other/TPLSD submodule
        "min_length": 15,
        "max_num_lines": None,
        "force_num_lines": False,
        "tps_thresh": 0.25,
        "tps_lmbd": 0.5,
    }

    required_data_keys = ["image"]

    @staticmethod
    def _init_submodule(submodule_path):
        """Run git submodule update --init for a submodule if it is empty or missing."""
        path = Path(submodule_path)
        if path.is_dir() and any(path.iterdir()):
            return  # already populated
        print(f"Initializing submodule at {path} ...")
        subprocess.run(
            ["git", "submodule", "update", "--init", str(path.relative_to(_REPO_ROOT))],
            cwd=str(_REPO_ROOT),
            check=True,
            timeout=120,
        )

    def _get_tplsd_root(self):
        root = self.conf.tplsd_root
        if root is None:
            root = _DEFAULT_TPLSD_ROOT
        root = Path(root).expanduser().resolve()
        if not root.is_dir() or not any(root.iterdir()):
            # Auto-init the submodule when using the default path
            if self.conf.tplsd_root is None:
                self._init_submodule(root)
            if not root.is_dir() or not any(root.iterdir()):
                raise FileNotFoundError(
                    f"TP-LSD repo not found at {root}. "
                    "Run: git submodule update --init other/TPLSD  "
                    "or set extractor.tplsd_root to your TP-LSD path."
                )
        return root

    def load_tplsd_model(self, ckpt_path, device):
        root = self._get_tplsd_root()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        # Add DCNv2 (other/dcnv2) to sys.path so the _ext native module is importable
        if not _DCNV2_PATH.is_dir() or not any(_DCNV2_PATH.iterdir()):
            self._init_submodule(_DCNV2_PATH)
        if not _DCNV2_PATH.is_dir():
            raise FileNotFoundError(
                f"DCNv2 submodule not found at {_DCNV2_PATH}. "
                "Run: git submodule update --init other/dcnv2"
            )
        ext_files = list(_DCNV2_PATH.glob("_ext*.so")) + list(_DCNV2_PATH.glob("_ext*.pyd"))
        if not ext_files:
            raise ImportError(
                f"DCNv2 extension not built at {_DCNV2_PATH}. "
                f"To build: cd {_DCNV2_PATH} && python setup.py build_ext --inplace"
            )
        dcnv2_str = str(_DCNV2_PATH)
        if dcnv2_str not in sys.path:
            sys.path.insert(0, dcnv2_str)

        try:
            from modeling.TP_Net import Res160, Res320
            from utils.reconstruct import TPS_line
            from utils.utils import load_model
        except ImportError as e:
            raise ImportError(
                f"Failed to import TP-LSD modules from {root}. "
                f"Original error: {e}"
            ) from e

        head = {"center": 1, "dis": 4, "line": 1}
        var = self.conf.tplsd_variant
        if var == "tp320":
            model = Res320(task_dim=head)
            self._in_res = (320, 320)
        elif var == "tplite":
            model = Res160(task_dim=head, size=320)
            self._in_res = (320, 320)
        elif var == "tp512":
            model = Res320(task_dim=head)
            self._in_res = (512, 512)
        else:
            raise ValueError(f"tplsd_variant must be tp320, tplite, or tp512, got {var}")

        model = load_model(model, str(ckpt_path))
        self._TPS_line = TPS_line
        return model.eval().to(device)

    def _init(self, conf):
        if self.conf.force_num_lines:
            assert self.conf.max_num_lines is not None, "Missing max_num_lines parameter"
        root = self._get_tplsd_root()
        var = self.conf.tplsd_variant
        ckpt_name = {"tp320": "Res320.pth", "tplite": "Res160.pth", "tp512": "Res512.pth"}[var]
        ckpt = root / "pretraineds" / ckpt_name
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"TP-LSD weights not found at {ckpt}. "
                f"Download {ckpt_name} from TP-LSD releases/pretraineds and put it in {ckpt.parent}. "
                f"See https://github.com/Siyuada7/TP-LSD for download links."
            )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = self.load_tplsd_model(ckpt, device)
        self.set_initialized(True)

    def _preprocess_tplsd(self, img_bgr, in_res):
        """
        Preprocess image for TP-LSD: resize and apply HSV V-channel enhancement.
        
        This matches the official TP-LSD preprocessing from demo_line.py:
        - Resize to target resolution
        - Convert to HSV
        - Apply V-channel enhancement (downscale, blur, upscale, blur, subtract from original)
        - Convert back to BGR and normalize to [0, 1]
        
        Args:
            img_bgr: BGR image [H, W, 3] as numpy array [0, 255]
            in_res: Target resolution (H, W) tuple
            
        Returns:
            Preprocessed image [H, W, 3] as float32 [0, 1], and actual (H, W)
        """
        # Resize to target resolution (matches demo_line.py line 296)
        inp = cv2.resize(img_bgr, (in_res[1], in_res[0]), interpolation=cv2.INTER_AREA)
        H, W, C = inp.shape
        
        # Convert to HSV and extract V channel (matches demo_line.py lines 298-299)
        hsv = cv2.cvtColor(inp, cv2.COLOR_BGR2HSV)
        imgv0 = hsv[..., 2]
        
        # V-channel enhancement: downscale, blur, upscale, blur (matches demo_line.py lines 300-303)
        imgv = cv2.resize(imgv0, (0, 0), fx=1.0 / 4, fy=1.0 / 4, interpolation=cv2.INTER_LINEAR)
        imgv = cv2.GaussianBlur(imgv, (5, 5), 3)
        imgv = cv2.resize(imgv, (W, H), interpolation=cv2.INTER_LINEAR)
        imgv = cv2.GaussianBlur(imgv, (5, 5), 3)
        
        # Subtract blurred from original and add 127.5 (matches demo_line.py lines 305-306)
        imgv1 = imgv0.astype(np.float32) - imgv + 127.5
        imgv1 = np.clip(imgv1, 0, 255).astype(np.uint8)
        hsv[..., 2] = imgv1
        
        # Convert back to BGR and normalize (matches demo_line.py lines 308-310)
        inp = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return inp.astype(np.float32) / 255.0, H, W

    def _forward(self, data):
        """
        Forward pass for TP-LSD line detection.
        
        Args:
            data: dict with 'image' tensor [B, C, H, W] in RGB format [0, 1]
            
        Returns:
            dict with:
                - 'lines': [B, N, 2, 2] tensor (or list) with line endpoints (x, y)
                - 'line_scores': [B, N] tensor (or list) with scores
                - 'valid_lines': [B, N] bool tensor (or list) indicating valid lines
        """
        image = data["image"]
        lines, line_scores, valid_lines = [], [], []
        in_res = self._in_res
        TPS_line = self._TPS_line
        device = next(self.net.parameters()).device

        for i in range(len(image)):
            im = image[i]
            if im.shape[0] == 3:
                # CHW [0,1] RGB -> HWC BGR [0,255]
                rgb = (im.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            else:
                gray = (im[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            H_img, W_img = img_bgr.shape[0], img_bgr.shape[1]
            inp, H, W = self._preprocess_tplsd(img_bgr, in_res)
            inp_t = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

            # Forward pass (matches demo_line.py lines 312-313)
            with torch.no_grad():
                outputs = self.net(inp_t)
            
            # Get last output (model returns list with single dict) - matches demo_line.py line 286
            output = outputs[-1]
            
            # Extract line segments using TPS_line reconstruction (matches demo_line.py line 287)
            # Default thresholds in official demo: thresh=0.25, lmbd=0.5
            segs, _, _, _, _ = TPS_line(
                output,
                thresh=self.conf.tps_thresh,
                lmbd=self.conf.tps_lmbd,
                H=H,
                W=W,
            )
            # Scale coordinates from resized (H, W) to original image size
            # Matches demo_line.py lines 288-291
            if len(segs) > 0:
                segs = segs.astype(np.float32)
                W_ = W_img / W
                H_ = H_img / H
                segs[:, [0, 2]] *= W_  # Scale x coordinates
                segs[:, [1, 3]] *= H_  # Scale y coordinates
                # Reshape from [N, 4] (x1, y1, x2, y2) to [N, 2, 2] (start, end)
                line_pred = segs.reshape(-1, 2, 2)
            else:
                line_pred = np.zeros((0, 2, 2), dtype=np.float32)

            lengths = np.linalg.norm(line_pred[:, 1] - line_pred[:, 0], axis=1)
            to_keep = lengths >= self.conf.min_length
            line_pred = line_pred[to_keep]
            lengths = lengths[to_keep]
            scores = np.sqrt(lengths)

            if self.conf.max_num_lines is not None:
                order = np.argsort(-scores)[: self.conf.max_num_lines]
                line_pred = line_pred[order]
                scores = scores[order]

            n = len(line_pred)
            valid_mask = np.ones(n, dtype=bool)
            if self.conf.force_num_lines and self.conf.max_num_lines is not None:
                pad = self.conf.max_num_lines - n
                if pad > 0:
                    line_pred = np.concatenate(
                        [line_pred, np.zeros((pad, 2, 2), dtype=np.float32)], axis=0
                    )
                    scores = np.concatenate([scores, np.zeros(pad, dtype=np.float32)], axis=0)
                    valid_mask = np.concatenate([valid_mask, np.zeros(pad, dtype=bool)], axis=0)

            lines.append(line_pred)
            line_scores.append(scores)
            valid_lines.append(valid_mask)

        if len(image) == 1 or self.conf.force_num_lines:
            lines = (
                torch.from_numpy(np.stack(lines, axis=0).astype(np.float32))
                .to(image.device)
                .float()
            )
            line_scores = (
                torch.from_numpy(np.stack(line_scores, axis=0).astype(np.float32))
                .to(image.device)
                .float()
            )
            valid_lines = (
                torch.from_numpy(np.stack(valid_lines, axis=0).astype(np.uint8))
                .to(image.device)
                .bool()
            )

        return {"lines": lines, "line_scores": line_scores, "valid_lines": valid_lines}

    def loss(self, pred, data):
        raise NotImplementedError
