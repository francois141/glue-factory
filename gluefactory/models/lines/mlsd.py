import subprocess

import numpy as np
import torch

from gluefactory.models.mlsd_backbone import MobileV2_MLSD_Large, MobileV2_MLSD_Tiny, pred_lines

from ...settings import DATA_PATH
from ..base_model import BaseModel

# Weights from mlsd_pytorch (user may need to download if not present)
MLSD_WEIGHTS_URL = {
    "tiny": "https://github.com/lhwcv/mlsd_pytorch/raw/main/models/mlsd_tiny_512_fp32.pth",
    "large": "https://github.com/lhwcv/mlsd_pytorch/raw/main/models/mlsd_large_512_fp32.pth",
}


class MLSD(BaseModel):
    default_conf = {
        "mlsd_size": "tiny",  # "tiny" | "large"
        "input_size": 512,
        "score_thr": 0.1,
        "dist_thr": 20.0,
        "min_length": 15,
        "max_num_lines": None,
        "force_num_lines": False,
    }

    required_data_keys = ["image"]

    def load_mlsd_model(self, ckpt_path, device="cuda"):
        size = self.conf.mlsd_size
        if size == "tiny":
            model = MobileV2_MLSD_Tiny()
        elif size == "large":
            model = MobileV2_MLSD_Large()
        else:
            raise ValueError(f"mlsd_size must be 'tiny' or 'large', got {size}")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        try:
            model.load_state_dict(state["model_state"], strict=True)
        except Exception:
            model.load_state_dict(state, strict=True)
        return model.eval().to(device)

    def _init(self, conf):
        if self.conf.force_num_lines:
            assert self.conf.max_num_lines is not None, "Missing max_num_lines parameter"
        ckpt = DATA_PATH / "weights" / f"mlsd_{self.conf.mlsd_size}_512_fp32.pth"
        if not ckpt.is_file():
            self.download_model(ckpt)
        
        # Test CUDA actually works before using it (handles broken CUDA runtime)
        device = "cpu"
        if torch.cuda.is_available():
            try:
                # Actually test CUDA with a real operation
                test_tensor = torch.zeros(1, device="cuda")
                test_result = test_tensor + 1
                del test_tensor, test_result
                torch.cuda.empty_cache()
                device = "cuda"
            except Exception:
                # CUDA reports available but runtime fails - use CPU
                device = "cpu"
        
        self.net = self.load_mlsd_model(ckpt, device)
        self.set_initialized()

    def download_model(self, path):
        size = self.conf.mlsd_size
        url = MLSD_WEIGHTS_URL.get(size)
        if not url:
            raise FileNotFoundError(
                f"MLSD weights for '{size}' not found at {path}. "
                "Download from https://github.com/lhwcv/mlsd_pytorch and put "
                f"mlsd_{size}_512_fp32.pth into {path.parent}."
            )
        if not path.parent.is_dir():
            path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading M-LSD ({size}) weights...")
        subprocess.run(["wget", url, "-O", str(path)], check=True)

    def _forward(self, data):
        image = data["image"]
        lines, line_scores, valid_lines = [], [], []
        device = next(self.net.parameters()).device
        input_shape = (self.conf.input_size, self.conf.input_size)

        for i in range(len(image)):
            im = image[i]
            if im.shape[0] == 3:
                # CHW -> HWC, [0,1] -> [0,255] RGB
                img_np = (im.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            else:
                # grayscale: repeat to RGB
                gray = (im[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                img_np = np.stack([gray, gray, gray], axis=-1)

            segs = pred_lines(
                img_np,
                self.net,
                input_shape=input_shape,
                score_thr=self.conf.score_thr,
                dist_thr=self.conf.dist_thr,
                device=device,
            )
            # segs [N, 4] -> [N, 2, 2] (x,y) for the two endpoints
            if len(segs) == 0:
                line_pred = np.zeros((0, 2, 2), dtype=np.float32)
                scores = np.zeros(0, dtype=np.float32)
            else:
                line_pred = segs.reshape(-1, 2, 2)
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

        if len(image) == 1:
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
