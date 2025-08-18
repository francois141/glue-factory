import numpy as np
import torch
import time
from joblib import Parallel, delayed
from pytlsd import lsd, lsd_from_points, lsd_opt
from faster_pytlsd import lsd as fast_lsd
from faster_pytlsd import params_lsd
import torchvision.transforms as T

from ..base_model import BaseModel


class LSD(BaseModel):
    default_conf = {
        "min_length": 15,
        "max_num_lines": None,
        "force_num_lines": False,
        "n_jobs": 4,
        "run_type": "default",
    }
    required_data_keys = ["image"]

    def _init(self, conf):
        if self.conf.force_num_lines:
            assert (
                self.conf.max_num_lines is not None
            ), "Missing max_num_lines parameter"

    def compute_gradient_2d_noborder(self, in_tensor: torch.Tensor) -> torch.Tensor:

        # GaussianBlur: kernel size must be odd and positive
        blur = T.GaussianBlur(kernel_size=7, sigma=0.6)

        # Apply blur
        in_tensor = blur(in_tensor.unsqueeze(0))[0]

        H, W = in_tensor.shape
        com1 = in_tensor[1:-1, 1:-1] - in_tensor[:-2, :-2]
        com2 = in_tensor[:-2, 1:-1] - in_tensor[1:-1, :-2]

        gx = com1 + com2
        gy = com1 - com2
        norm2 = gx * gx + gy * gy
        norm = torch.sqrt(norm2 / 4.0)

        out = torch.zeros_like(in_tensor).to(torch.float32)
        out[1:-1, 1:-1] = norm

        # Compute angle where norm > threshold
        threshold = 5.2262518595055063
        mask = norm > threshold

        # atan2(-gx, gy) for pixels where norm > threshold
        angle_vals = np.atan2(-gx[mask], gy[mask])

        # Insert angles back in output angle tensor
        out_angle = np.zeros_like(in_tensor)
        out_angle[1:-1, 1:-1][mask] = angle_vals

        return out, out_angle

    def extract_points(self, gradnorm: torch.Tensor):
        gradnorm = gradnorm.cpu().numpy()

        positions = np.argwhere(gradnorm > 0)  

        intensities = gradnorm[positions[:, 0], positions[:, 1]]

        sorted_indices = np.argsort(-intensities)
        positions_sorted = positions[sorted_indices]

        # Return keypoints sorted by intensity
        return positions_sorted[:, [1, 0]].astype(int).flatten().reshape(-1, 2)

    def detect_lines(self, img):
        start = time.perf_counter()
        # Run LSD
        if 'search' in self.conf and self.conf.search:
            segs = params_lsd(
                img,
                scale=self.conf.scale,
                sigma_scale=self.conf.sigma_scale,
                density_th=0.0,
                quant=self.conf.quant,
                ang_th=self.conf.angle_th,
                with_gaussian=self.conf.with_gaussian
            )
        elif self.conf.run_type == "default":
            segs = lsd(img)
        elif self.conf.run_type == "fast":
            segs = fast_lsd(img)
        elif self.conf.run_type == "optimal":
            segs = lsd_opt(img)
        elif self.conf.run_type == "points":
            gradient, angles = self.compute_gradient_2d_noborder(torch.tensor(img))
            interests_points = self.extract_points(gradient)
            segs = lsd_from_points(
                img,
                interests_points,
                scale=1.0,
                gradnorm=gradient,
                gradangle=angles
            )

        end = time.perf_counter()
        # Convert latency in milliseconds
        latency = (end - start) * 1_000

        # Filter out keylines that do not meet the minimum length criteria
        lengths = np.linalg.norm(segs[:, 2:4] - segs[:, 0:2], axis=1)
        to_keep = lengths >= self.conf.min_length
        segs, lengths = segs[to_keep], lengths[to_keep]

        # Keep the best lines
        scores = segs[:, -1] * np.sqrt(lengths)
        segs = segs[:, :4].reshape(-1, 2, 2)
        indices = np.argsort(-scores)
        if self.conf.max_num_lines is not None:
            indices = indices[: self.conf.max_num_lines]
            segs = segs[indices]
            scores = scores[indices]

        # Pad if necessary
        n = len(segs)
        valid_mask = np.ones(n, dtype=bool)
        if self.conf.force_num_lines:
            pad = self.conf.max_num_lines - n
            segs = np.concatenate(
                [segs, np.zeros((pad, 2, 2), dtype=np.float32)], axis=0
            )
            scores = np.concatenate([scores, np.zeros(pad, dtype=np.float32)], axis=0)
            valid_mask = np.concatenate([valid_mask, np.zeros(pad, dtype=bool)], axis=0)

        return segs, scores, valid_mask, latency

    def _forward(self, data):
        # Convert to the right data format
        image = data["image"]
        if image.shape[1] == 3:
            # Convert to grayscale
            scale = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            image = (image * scale).sum(1, keepdim=True)
        device = image.device
        b_size = len(image)
        image = np.uint8(image.squeeze(1).cpu().numpy() * 255)

        # LSD detection in parallel
        if b_size == 1:
            lines, line_scores, valid_lines, latencies = self.detect_lines(image[0])
            lines = [lines]
            line_scores = [line_scores]
            valid_lines = [valid_lines]
            latencies = [latencies]
        else:
            lines, line_scores, valid_lines, latencies = zip(
                *Parallel(n_jobs=self.conf.n_jobs)(
                    delayed(self.detect_lines)(img) for img in image
                )
            )

        # Batch if possible
        if b_size == 1 or self.conf.force_num_lines:
            lines = torch.tensor(lines, dtype=torch.float, device=device)
            line_scores = torch.tensor(line_scores, dtype=torch.float, device=device)
            valid_lines = torch.tensor(valid_lines, dtype=torch.bool, device=device)

        return {"lines": lines, "line_scores": line_scores, "valid_lines": valid_lines, "latencies": latencies}

    def loss(self, pred, data):
        raise NotImplementedError
