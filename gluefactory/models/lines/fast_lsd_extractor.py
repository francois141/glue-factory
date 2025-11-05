import numpy as np
import torch
from faster_pytlsd import lsd as fast_lsd
from pytlsd import lsd, lsd_df, lsd_from_points, lsd_opt

from gluefactory.models.lines.line_refinement import merge_lines
from gluefactory.utils.image import (
    compute_image_grad,
    extract_all_points_sorted_by_gradient,
)

from ..base_model import BaseModel


class FastLSDLineExtractor(BaseModel):
    """
    This is meant to be a simple wrapper to use LSD or fast LSD in the JPL pipeline (joint_point_line_extractor.py)
    """

    default_conf = {
        "name": "lines.fast_lsd_extractor",
        "min_length": -1,
        "max_num_lines": None,
        "force_num_lines": False,
        "use_img_grad_angle": True,  # Dont use the angle-field but use the image gradient as surrogate
        "merge": False,
        "grad_nfa": True,
        "filtering": "normal",
        "grad_thresh": 3,
        "return_line_descriptors": False,
        "trainable": False,
        "sigma": 0.6,
        "threshold_value": 5.2262518595055063,
        "kernel_size": 7,
        "lsd_type": "point", # point is the best one currently
    }
    required_data_keys = ["image"]

    def _init(self, conf):
        if self.conf.force_num_lines:
            assert (
                self.conf.max_num_lines is not None
            ), "Missing max_num_lines parameter"
        # currently line descriptors arte not implemented
        # TODO: remove not implemented option
        if self.conf.return_line_descriptors:
            raise NotImplementedError(
                "Line descriptors are not implemented yet for FasterLSD"
            )

        # Currently we have 5 types of lsd entries
        assert self.conf.lsd_type in ["default", "old", "best", "fast", "point"]

    def detect_lines(self, img, df, line_level=None):
        """
        detect lines in one image.
        Args:
            img: image as numpy array
            df: denormalized distance field as numpy array
            line_level: line anglefield / line level as numpy array. Not needed if conf.use_img_grad_angle is True.
        Returns: numpy array containing lines as (x1, y1 \\ x2, y2) tuples so of shape (n_lines, 2, 2)
        """
        # Run LSD
        img_grad_angle = None
        gradnorm = torch.clamp(5.0 - df, min=0.0).to(torch.float64)

        # Lsd types: default, old, best, fast, point
        if self.conf.lsd_type == "default":
            lines = lsd_df(
                img.detach().cpu().numpy().astype(np.float64),
                scale=1.0,
                gradnorm=gradnorm.detach().cpu().numpy(),
                grad_nfa=False,
            )[:, :4].reshape(-1, 2, 2)
        elif self.conf.lsd_type == "old":
            # Compute grad angle old style
            img_grad_angle = compute_image_grad(img, 7, self.conf.sigma)
            angle = (img_grad_angle - torch.pi / 2) % (2 * torch.pi)
            angle[gradnorm < self.conf.grad_thresh] = -1024
            lines = lsd(
                img.detach().cpu().numpy().astype(np.float64),
                scale=1.0,
                gradnorm=gradnorm.detach().cpu().numpy(),
                gradangle=angle.detach().cpu().numpy(),
                grad_nfa=False,
            )[:, :4].reshape(-1, 2, 2)
        elif self.conf.lsd_type == "best":
            gradient, angle = self.compute_gradient_2d_noborder(img)
            lines = lsd_opt(
                img.cpu().detach().numpy().astype(np.float64),
                scale=1.0,
                gradnorm=gradnorm.detach().cpu().numpy(),
                gradangle=angle.detach().cpu().numpy(),
                grad_nfa=False,
            )[:, :4].reshape(-1, 2, 2)
        elif self.conf.lsd_type == "fast":
            gradient, angle = self.compute_gradient_2d_noborder(img)
            lines = fast_lsd(
                img.detach().cpu().numpy().astype(np.float64),
                scale=1.0,
                gradnorm=gradnorm.detach().cpu().numpy(),
                gradangle=angle.detach().cpu().numpy(),
                grad_nfa=self.conf.grad_nfa,
            )[:, :4].reshape(-1, 2, 2)
        elif self.conf.lsd_type == "point":
            interests_points = extract_all_points_sorted_by_gradient(gradnorm)
            img_grad_angle = compute_image_grad(img, 7, self.conf.sigma)
            angle = (img_grad_angle - torch.pi / 2) % (2 * torch.pi)
            angle[gradnorm < self.conf.grad_thresh] = -1024
            lines = lsd_from_points(
                img.detach().cpu().numpy().astype(np.float64),
                interests_points.detach().cpu().numpy().astype(np.float64),
                scale=1.0,
                gradnorm=gradnorm.detach().cpu().numpy(),
                gradangle=angle.detach().cpu().numpy(),
                grad_nfa=False,
            )[:, :4].reshape(-1, 2, 2)

        lines = torch.tensor(lines)

        # Now perform optional min length filtering and apply force num lines if needed
        if self.conf.min_length:
            lengths = torch.norm(lines[:, 0] - lines[:, 1], dim=1)
            to_keep = lengths >= self.conf.min_length
            lines, lengths = lines[to_keep], lengths[to_keep]

        # Keep the best lines (best lines are the shortest ones)
        if self.conf.max_num_lines is not None:
            scores = torch.sqrt(lengths)
            lines = lines[:, :4].reshape(-1, 2, 2)
            indices = torch.argsort(-scores)
            indices = indices[: self.conf.max_num_lines]
            lines = lines[indices]

        if self.conf.merge:
            lines = merge_lines(lines, thresh=4, overlap_thresh=0)

        n = len(lines)
        valid_mask = torch.ones(n, dtype=bool, device=lines.device)

        # Pad if necessary
        if self.conf.force_num_lines:
            pad = self.conf.max_num_lines - n
            if pad > 0:
                pad_lines = torch.zeros(
                    (pad, 2, 2), dtype=lines.dtype, device=lines.device
                )
                lines = torch.cat([lines, pad_lines], dim=0)

                pad_mask = torch.zeros(pad, dtype=torch.bool, device=lines.device)
                valid_mask = torch.cat([valid_mask, pad_mask], dim=0)

        return {"lines": lines, "valid_lines": valid_mask}

    def _forward(self, data):
        """
        Perform forward pass on the data. Supports batched data.
        Args:
            data: dictionary containing the data. Must contain the following keys: image, line_angle_field, line_distance_field.
        Returns: dict containing the lines and valid_lines for each image.
        """
        # Convert to the right data format
        image = data["image"]
        line_level = data["line_anglefield"]
        line_df_denormalized = data["line_distancefield"]

        # preprocess input to lsd
        image = (image[:, 0] * 255).to(torch.uint8)

        def process_one(img, df, ll):
            line_pred = self.detect_lines(img, df, ll)
            return line_pred["lines"], line_pred["valid_lines"]

        results = []

        if len(image) == 1:
            results.append(
                process_one(image[0], line_df_denormalized[0], line_level[0])
            )
        else:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(process_one, img, df, ll)
                    for img, df, ll in zip(image, line_df_denormalized, line_level)
                ]
                for f in futures:
                    results.append(f.result())

        # Unpack results
        lines, valid_lines = zip(*results)

        return {"lines": list(lines), "valid_lines": list(valid_lines)}

    def loss(self, pred, data):
        raise NotImplementedError
