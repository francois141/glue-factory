from typing import Optional

import numpy as np
import torch
from faster_pytlsd import lsd as fast_lsd
from pytlsd import lsd

from gluefactory.models.lines.line_refinement import filter_outlier_lines, merge_lines
from gluefactory.models.lines.line_utils import preprocess_angle
from gluefactory.utils.image import compute_image_grad

from ..base_model import BaseModel


class FastLSDLineExtractor(BaseModel):
    """
    This is meant to be a simple wrapper to use LSD or fast LSD in the JPL pipeline (joint_point_line_extractor.py)
    """

    default_conf = {
        "name": "lines.fast_lsd_extractor",
        "min_length": 15,
        "max_num_lines": None,
        "force_num_lines": False,
        "use_img_grad_angle": True,  # Dont use the angle-field but use the image gradient as surrogate
        "merge": False,
        "grad_nfa": True,
        "filtering": "normal",
        "grad_thresh": 3,
        "faster_lsd": True,
        "return_line_descriptors": False,
        "trainable": False,
    }
    required_data_keys = ["image"]

    def _init(self, conf):
        if self.conf.force_num_lines:
            assert (
                self.conf.max_num_lines is not None
            ), "Missing max_num_lines parameter"
        # currently line descriptors arte not implemented
        if self.conf.return_line_descriptors:
            raise NotImplementedError(
                "Line descriptors are not implemented yet for FasterLSD"
            )

    def detect_lines(
        self, img, df, line_level = None
    ):
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

        if self.conf.faster_lsd:
            lines = fast_lsd(
                img.cpu().numpy().astype(np.float64),
                scale=1.0,
                gradnorm=gradnorm.cpu().numpy(),
                grad_nfa=self.conf.grad_nfa,
            )[:, :4].reshape(-1, 2, 2)
        else:
            lines = lsd(
                img.cpu().numpy().astype(np.float64),
                scale=1.0,
                gradnorm=gradnorm.cpu().numpy(),
                grad_nfa=self.conf.grad_nfa,
            )[:, :4].reshape(-1, 2, 2)


        lines = torch.tensor(lines)

        # Now perform optional min length filtering and apply force num lines if needed
        lengths = torch.norm(lines[:, 0] - lines[:, 1], dim=1)
        to_keep = lengths >= self.conf.min_length
        lines, lengths = lines[to_keep], lengths[to_keep]

        # Keep the best lines (best lines are the shortest ones)
        scores = torch.sqrt(lengths)
        lines = lines[:, :4].reshape(-1, 2, 2)
        indices = torch.argsort(-scores)
        if self.conf.max_num_lines is not None:
            indices = indices[: self.conf.max_num_lines]
            lines = lines[indices]

        if self.conf.merge:
            lines = merge_lines(
                lines, thresh=4, overlap_thresh=0
            )

        # Pad if necessary
        n = len(lines)
        valid_mask = torch.ones(n, dtype=bool, device=lines.device)
        if self.conf.force_num_lines:
            pad = self.conf.max_num_lines - n
            if pad > 0:
                pad_lines = torch.zeros((pad, 2, 2), dtype=lines.dtype, device=lines.device)
                lines = torch.cat([lines, pad_lines], dim=0)

                pad_mask = torch.zeros(pad, dtype=torch.bool, device=lines.device)
                valid_mask = torch.cat([valid_mask, pad_mask], dim=0)

        return {"lines": lines, "valid_lines": valid_mask}

    def _forward(self, data):
        """
        Perform forward pass on the data. Supports batched data.
        Args:
            data: dictionary containing the data. Must contain the following keys: image, line_angle_field, line_distance_field.
        Returns: a list of tensors, containing the lines for each image: shape: [N_images x (n_lines, 2, 2)]
        """
        # Convert to the right data format
        image = data["image"]
        line_level = data["line_anglefield"]
        line_df_denormalized = data["line_distancefield"]

        # preprocess input to lsd
        image = (image[:, 0] * 255).to(torch.uint8)
        lines = []

        # valid lines contain a line mask indicating which lines were actually predicted and which are padding
        # applied to enable batching via conf.force_num_lines
        valid_lines = []
        for img, df, ll in zip(image, line_df_denormalized, line_level):
            line_pred = self.detect_lines(img, df, line_level)
            lines.append(line_pred["lines"])
            valid_lines.append(line_pred["valid_lines"])
        # Here a list of lines for each img is returned.
        outputs = {"lines": lines, "valid_lines": valid_lines}
        return outputs

    def loss(self, pred, data):
        raise NotImplementedError
