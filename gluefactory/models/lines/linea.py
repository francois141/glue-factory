import subprocess
import sys
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F

from ...settings import DATA_PATH
from ..base_model import BaseModel


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_LINEA_ROOT = _REPO_ROOT / "external" / "LINEA"

LINEA_WEIGHTS_URL = {
    "n": "https://github.com/SebastianJanampa/storage/releases/download/LINEA/linea_hgnetv2_n.pth",
    "s": "https://github.com/SebastianJanampa/storage/releases/download/LINEA/linea_hgnetv2_s.pth",
    "m": "https://github.com/SebastianJanampa/storage/releases/download/LINEA/linea_hgnetv2_m.pth",
    "l": "https://github.com/SebastianJanampa/storage/releases/download/LINEA/linea_hgnetv2_l.pth",
}


class LINEA(BaseModel):
    default_conf = {
        "linea_variant": "n",
        "linea_root": None,
        "input_size": 640,
        "score_threshold": 0.4,
        "min_length": 15,
        "max_num_lines": None,
        "force_num_lines": False,
        "save_line_images": False,
        "line_image_path": "image_line.jpg",
        "line_image_width": 2,
    }
    required_data_keys = ["image"]

    def _get_linea_root(self):
        root = self.conf.linea_root
        root = _DEFAULT_LINEA_ROOT if root is None else Path(root).expanduser()
        root = root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(
                f"LINEA repo not found at {root}. "
                "Run ./fetch_external.sh or set model.linea_root to a LINEA checkout."
            )
        return root

    def _add_linea_to_path(self, root):
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        for module_name in ("models", "util"):
            module = sys.modules.get(module_name)
            if module is None or not hasattr(module, "__file__"):
                continue
            module_file = Path(module.__file__).resolve()
            if root not in module_file.parents:
                raise ImportError(
                    f"Cannot import LINEA because Python module '{module_name}' "
                    f"is already loaded from {module_file}. Start a fresh process "
                    "or load LINEA before other packages using that top-level name."
                )

    def _import_linea_modules(self, root):
        self._add_linea_to_path(root)
        try:
            from util.slconfig import SLConfig
            from models.registry import MODULE_BUILD_FUNCS
        except ImportError as exc:
            raise ImportError(
                "Failed to import LINEA. Run ./fetch_external.sh to install LINEA "
                "and its requirements."
            ) from exc
        return SLConfig, MODULE_BUILD_FUNCS

    def _build_linea_model(self, root, ckpt_path, device, slconfig, build_funcs):
        config_path = root / "configs" / "linea" / f"linea_hgnetv2_{self.conf.linea_variant}.py"
        if not config_path.is_file():
            raise FileNotFoundError(f"LINEA config not found at {config_path}")

        cfg = slconfig.fromfile(str(config_path))
        if "HGNetv2" in cfg.backbone:
            cfg.pretrained = False
        cfg.multiscale = None

        build_func = build_funcs.get(cfg.modelname)
        if build_func is None:
            raise RuntimeError(f"LINEA model builder '{cfg.modelname}' is not registered")
        model, postprocessor = build_func(cfg)

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            state = checkpoint["model"]
        else:
            state = checkpoint
        model.load_state_dict(state)

        model = model.deploy().eval().to(device)
        postprocessor = postprocessor.deploy().eval().to(device)
        return model, postprocessor

    def _init(self, conf):
        if self.conf.force_num_lines:
            assert self.conf.max_num_lines is not None, "Missing max_num_lines parameter"
        if self.conf.linea_variant not in LINEA_WEIGHTS_URL:
            raise ValueError(
                f"linea_variant must be one of {sorted(LINEA_WEIGHTS_URL)}, "
                f"got {self.conf.linea_variant}"
            )

        root = self._get_linea_root()
        slconfig, build_funcs = self._import_linea_modules(root)
        ckpt = DATA_PATH / "weights" / f"linea_hgnetv2_{self.conf.linea_variant}.pth"
        if not ckpt.is_file():
            self.download_model(ckpt)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net, self.postprocessor = self._build_linea_model(
            root, ckpt, device, slconfig, build_funcs
        )
        self.set_initialized(True)

    def download_model(self, path):
        url = LINEA_WEIGHTS_URL[self.conf.linea_variant]
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading LINEA-{self.conf.linea_variant.upper()} weights...")
        subprocess.run(["wget", url, "-O", str(path)], check=True)

    def _prepare_images(self, image):
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        elif image.shape[1] != 3:
            raise ValueError(f"Expected 1 or 3 image channels, got {image.shape[1]}")

        image = F.interpolate(
            image,
            size=(self.conf.input_size, self.conf.input_size),
            mode="bilinear",
            align_corners=False,
        )

        # They seems to normalize here
        # https://github.com/SebastianJanampa/LINEA/blob/master/datasets/coco.py
        mean = image.new_tensor([0.538, 0.494, 0.453]).view(1, 3, 1, 1)
        std = image.new_tensor([0.257, 0.263, 0.273]).view(1, 3, 1, 1)
        return (image - mean) / std

    def _filter_lines(self, lines, scores):
        if self.conf.score_threshold is not None:
            keep = scores > self.conf.score_threshold
            lines = lines[keep]
            scores = scores[keep]

        if len(lines) > 0 and self.conf.min_length is not None:
            lengths = torch.linalg.norm(lines[:, 1] - lines[:, 0], dim=1)
            keep = lengths >= self.conf.min_length
            lines = lines[keep]
            scores = scores[keep]

        if len(lines) > 0:
            order = torch.argsort(scores, descending=True)
            if self.conf.max_num_lines is not None:
                order = order[: self.conf.max_num_lines]
            lines = lines[order]
            scores = scores[order]

        valid_mask = torch.ones(len(lines), dtype=torch.bool, device=lines.device)
        if self.conf.force_num_lines:
            pad = self.conf.max_num_lines - len(lines)
            if pad > 0:
                lines = torch.cat(
                    [lines, torch.zeros((pad, 2, 2), dtype=lines.dtype, device=lines.device)],
                    dim=0,
                )
                scores = torch.cat(
                    [scores, torch.zeros(pad, dtype=scores.dtype, device=scores.device)],
                    dim=0,
                )
                valid_mask = torch.cat(
                    [valid_mask, torch.zeros(pad, dtype=torch.bool, device=valid_mask.device)],
                    dim=0,
                )

        return lines, scores, valid_mask

    def _save_line_image(self, image, lines):
        image_np = (
            image.detach()
            .cpu()
            .clamp(0, 1)
            .permute(1, 2, 0)
            .numpy()
        )
        if image_np.shape[-1] == 1:
            image_np = image_np.repeat(3, axis=-1)
        image_np = (image_np * 255).astype("uint8")
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        lines_np = lines.detach().cpu().numpy().round().astype("int32")
        for line in lines_np:
            pt1 = tuple(line[0].tolist())
            pt2 = tuple(line[1].tolist())
            cv2.line(
                image_bgr,
                pt1,
                pt2,
                color=(0, 0, 255),
                thickness=self.conf.line_image_width,
                lineType=cv2.LINE_AA,
            )

        output_path = Path(self.conf.line_image_path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image_bgr)

    def _forward(self, data):
        image = data["image"]
        device = next(self.net.parameters()).device
        orig_sizes = torch.tensor(
            [[image.shape[-1], image.shape[-2]]] * len(image),
            dtype=torch.float32,
            device=device,
        )
        model_input = self._prepare_images(image.to(device))

        with torch.no_grad():
            outputs = self.net(model_input)
            pred_lines, pred_scores = self.postprocessor(outputs, orig_sizes)

        lines, line_scores, valid_lines = [], [], []
        for line, score in zip(pred_lines, pred_scores):
            line = line.reshape(-1, 2, 2).to(image.device)
            score = score.to(image.device)
            line, score, valid_mask = self._filter_lines(line, score)
            lines.append(line)
            line_scores.append(score)
            valid_lines.append(valid_mask)

        if self.conf.save_line_images:
            for im, line, valid_mask in zip(image, lines, valid_lines):
                self._save_line_image(im, line[valid_mask])

        if len(image) == 1 or self.conf.force_num_lines:
            lines = torch.stack(lines, dim=0)
            line_scores = torch.stack(line_scores, dim=0)
            valid_lines = torch.stack(valid_lines, dim=0)

        return {"lines": lines, "line_scores": line_scores, "valid_lines": valid_lines}

    def loss(self, pred, data):
        raise NotImplementedError
    
