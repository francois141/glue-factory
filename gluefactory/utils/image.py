import collections.abc as collections
from pathlib import Path
from typing import Optional, Tuple

import cv2
import kornia
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
import torchvision.transforms as T
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from ..geometry.wrappers import Camera


class ImagePreprocessor:
    default_conf = {
        "resize": None,  # target edge length, None for no resizing
        "edge_divisible_by": None,
        "side": "long",
        "interpolation": "bilinear",
        "align_corners": None,
        "antialias": True,
        "square_pad": False,
        "add_padding_mask": False,
    }

    def __init__(self, conf) -> None:
        super().__init__()
        default_conf = OmegaConf.create(self.default_conf)
        OmegaConf.set_struct(default_conf, True)
        self.conf = OmegaConf.merge(default_conf, conf)

    def __call__(self, img: torch.Tensor, interpolation: Optional[str] = None) -> dict:
        """Resize and preprocess an image, return image and resize scale"""
        h, w = img.shape[-2:]
        size = h, w
        if self.conf.resize is not None:
            if interpolation is None:
                interpolation = self.conf.interpolation
            size = self.get_new_image_size(h, w)
            img = kornia.geometry.transform.resize(
                img,
                size,
                side=self.conf.side,
                antialias=self.conf.antialias,
                align_corners=self.conf.align_corners,
                interpolation=interpolation,
            )
        scale = torch.Tensor([img.shape[-1] / w, img.shape[-2] / h]).to(img)
        T = np.diag([scale[0], scale[1], 1])

        data = {
            "scales": scale,
            "image_size": np.array(size[::-1]),
            "transform": T,
            "original_image_size": np.array([w, h]),
        }
        if self.conf.square_pad:
            sl = max(img.shape[-2:])
            data["image"] = torch.zeros(
                *img.shape[:-2], sl, sl, device=img.device, dtype=img.dtype
            )
            data["image"][:, : img.shape[-2], : img.shape[-1]] = img
            if self.conf.add_padding_mask:
                data["padding_mask"] = torch.zeros(
                    *img.shape[:-3], 1, sl, sl, device=img.device, dtype=torch.bool
                )
                data["padding_mask"][:, : img.shape[-2], : img.shape[-1]] = True

        else:
            data["image"] = img
        return data

    def load_image(self, image_path: Path) -> dict:
        return self(load_image(image_path))

    def get_new_image_size(
        self,
        h: int,
        w: int,
    ) -> Tuple[int, int]:
        side = self.conf.side
        if isinstance(self.conf.resize, collections.Iterable):
            assert len(self.conf.resize) == 2
            return tuple(self.conf.resize)
        side_size = self.conf.resize
        aspect_ratio = w / h
        if side not in ("short", "long", "vert", "horz"):
            raise ValueError(
                f"side can be one of 'short', 'long', 'vert', and 'horz'. Got '{side}'"
            )
        if side == "vert":
            size = side_size, int(side_size * aspect_ratio)
        elif side == "horz":
            size = int(side_size / aspect_ratio), side_size
        elif (side == "short") ^ (aspect_ratio < 1.0):
            size = side_size, int(side_size * aspect_ratio)
        else:
            size = int(side_size / aspect_ratio), side_size

        if self.conf.edge_divisible_by is not None:
            df = self.conf.edge_divisible_by
            size = list(map(lambda x: int(x // df * df), size))
        return size


def read_image(path: Path, grayscale: bool = False) -> np.ndarray:
    """Read an image from path as RGB or grayscale"""
    if not Path(path).exists():
        raise FileNotFoundError(f"No image at path {path}.")
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise IOError(f"Could not read image at {path}.")
    if not grayscale:
        image = image[..., ::-1]
    return image


def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)


def load_image(path: Path, grayscale=False) -> torch.Tensor:
    image = read_image(path, grayscale=grayscale)
    return numpy_image_to_torch(image)


def grid_sample(
    image: torch.Tensor,
    coords: torch.Tensor,
    interpolation: str = "bilinear",
    align_corners: bool = False,
):
    assert image.dim() == coords.dim()
    is_batched = image.dim() == 4

    if is_batched:
        assert coords.dim() == 4
        return F.grid_sample(
            image.to(coords.device), coords, interpolation, align_corners=False
        )
    else:
        return F.grid_sample(
            image[None].to(coords.device),
            coords[None],
            mode=interpolation,
            align_corners=align_corners,
        )[0]


def get_pixel_grid(
    *,
    fmap: torch.Tensor | None = None,  # B x H X W X D
    camera: Camera | None = None,
    size: Optional[tuple[int, int]] = None,
    device: torch.device | None = None,
    dtype=torch.float32,
    normalized: bool = False,
) -> torch.Tensor:
    if fmap is None:
        if camera is None:
            if size is None:
                raise ValueError("Specify fmap, size, or camera")
            w, h = size
        else:
            w, h = camera.size.int()
            device = camera.device
            dtype = camera.dtype
    else:
        *_, h, w, _ = fmap.shape
        device = fmap.device
        dtype = fmap.dtype
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(w, dtype=dtype, device=device),
            torch.arange(h, dtype=dtype, device=device),
            indexing="xy",
        ),
        dim=-1,
    )
    if fmap.ndim == 4:
        b = fmap.shape[0]
        grid = grid[None].expand(b, -1, -1, -1)
    grid += 0.5
    if normalized:
        grid *= 2 / grid.new_tensor([w, h])
        grid -= 1
    elif fmap is not None and camera is not None:
        # In case fmaps are at a different image resolution
        grid *= camera.size / grid.new_tensor([w, h])
    return grid


def coords_to_image(coords):
    # ...HWC -> ...CHW
    return coords.transpose(-2, -1).transpose(-3, -2)


def image_to_coords(image):
    # ...CHW -> ...HWC
    return image.transpose(-3, -2).transpose(-2, -1)


def denormalize_coords(coords, hw: tuple[int, int] | None = None) -> torch.Tensor:
    """Denormalize coordinates from [-1, 1] to [0, H] or [0, W] (COLMAP)"""
    coords = coords.clone()
    if hw is None:
        hw = coords.shape[-3:-1]
    coords[..., 0] = (coords[..., 0] + 1) / 2 * (hw[1] - 1)
    coords[..., 1] = (coords[..., 1] + 1) / 2 * (hw[0] - 1)
    return coords


def normalize_coords(coords, hw: tuple[int, int] | None = None) -> torch.Tensor:
    """Normalize coordinates from [0, H] or [0, W] (COLMAP) to [-1, 1]"""
    coords = coords.clone()
    if hw is None:
        hw = coords.shape[-3:-1]
    coords[..., 0] = coords[..., 0] / (hw[1] - 1) * 2 - 1
    coords[..., 1] = coords[..., 1] / (hw[0] - 1) * 2 - 1
    return coords


def compute_image_grad(img: torch.Tensor, ksize: int = 7, sigma: float = 1.0):
    """
    Compute image gradient with Gaussian blur in PyTorch (using torchvision's GaussianBlur).
    Args:
        img (torch.Tensor): Input image tensor of shape (H, W) or (1, H, W).
        ksize (int): Gaussian kernel size (must be odd).
        sigma (float): Gaussian sigma.
    Returns:
        dx, dy, gradnorm, gradangle (torch.Tensor)
    """
    # ensure 3D tensor (C=1 for grayscale)
    if img.ndim == 2:
        img = img.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    elif img.ndim == 3:
        img = img.unsqueeze(0)  # (1,C,H,W)
    else:
        raise ValueError("Expected input shape (H,W) or (C,H,W)")

    # apply Gaussian blur (torchvision keeps same device/dtype)
    blur_img = TF.gaussian_blur(img.float(), kernel_size=ksize, sigma=sigma)

    # remove batch/channel for consistency with numpy code
    blur_img = blur_img.squeeze(0).squeeze(0)

    # compute gradients
    dx = torch.zeros_like(blur_img)
    dy = torch.zeros_like(blur_img)

    dx[:, 1:] = (blur_img[:, 1:] - blur_img[:, :-1]) / 2
    dx[1:, 1:] = dx[:-1, 1:] + dx[1:, 1:]

    dy[1:] = (blur_img[1:] - blur_img[:-1]) / 2
    dy[1:, 1:] = dy[1:, :-1] + dy[1:, 1:]

    gradangle = torch.atan2(dy, dx)

    return gradangle

def compute_lsd_image_gradient(in_tensor: torch.Tensor) -> torch.Tensor:
    device = in_tensor.device

    # GaussianBlur: kernel size must be odd and positive
    blur = T.GaussianBlur(kernel_size=7, sigma=0.6)

    # Apply blur
    in_tensor = blur(in_tensor.to(torch.float64).unsqueeze(0))[0]

    #in_tensor = in_tensor.int()
    com1 = in_tensor[1:, 1:] - in_tensor[:-1, :-1]
    com2 = in_tensor[:-1, 1:] - in_tensor[1:, :-1]

    gx = com1 + com2
    gy = com1 - com2
    norm2 = gx * gx + gy * gy
    norm = torch.sqrt(norm2 / 4.0)

    out = torch.zeros_like(in_tensor).to(device).to(torch.float64)
    out[1:, 1:] = norm

    # Compute angle where norm > threshold
    threshold = 5.2262518595055063
    mask = norm > threshold

    # atan2(-gx, gy) for pixels where norm > threshold
    angle_vals = torch.atan2(-gx[mask], gy[mask])
    epsilon = 1e-9
    angle_vals[torch.isclose(angle_vals, torch.tensor(torch.pi).to(torch.float64), atol=epsilon)] = -torch.pi

    # Insert angles back in output angle tensor
    out_angle = -1024 * (torch.ones(in_tensor.shape).to(device).to(torch.float64))
    out_angle[1:, 1:][mask] = angle_vals

    return out, out_angle

def extract_non_zero_points(gradnorm, amount=-1):
    positions = np.argwhere(gradnorm > 0)

    intensities = gradnorm[positions[:, 0], positions[:, 1]]

    sorted_indices = np.argsort(-intensities)
    positions_sorted = positions[sorted_indices]

    # Return keypoints sorted by intensity
    points = positions_sorted[:, [1, 0]].astype(int).flatten().reshape(-1, 2)

    if amount != -1:
        points = points[:amount]

    return points

def extract_non_zero_points_sorted_by_gradient(gradnorm: torch.Tensor, amount: int = -1) -> torch.Tensor:
    # Find positions where gradnorm > 0
    positions = torch.nonzero(gradnorm > 0, as_tuple=False)

    if positions.numel() == 0:
        return torch.empty((0, 2), dtype=torch.int64)

    # Get intensities at those positions
    intensities = gradnorm[positions[:, 0], positions[:, 1]]

    # Sort positions by intensity (descending)
    sorted_indices = torch.argsort(intensities, descending=True)
    positions_sorted = positions[sorted_indices]

    # Convert (row, col) -> (x, y) = (col, row)
    points = positions_sorted[:, [1, 0]]

    # Limit number of points if needed
    if amount != -1:
        points = points[:amount]

    return points


def extract_all_points_sorted_by_gradient(gradient: torch.Tensor, percentile: float = 75.0):
    """
    Extract points sorted by gradient intensity, then keep only those
    whose distance from the center is greater than the average distance
    and whose intensity is above a given percentile.
    
    Args:
        gradient (torch.Tensor): 2D gradient map [H, W].
        percentile (float): intensity percentile (e.g. 90 => top 10%).

    Returns:
        torch.Tensor: positions [N, 2] in (col, row) format.
    """
    # --- Create position grid ---
    rows = torch.arange(gradient.size(0), device=gradient.device)
    cols = torch.arange(gradient.size(1), device=gradient.device)
    grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")
    positions = torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=1)  # [num_points, 2]

    # --- Extract intensities ---
    intensities = gradient[positions[:, 0], positions[:, 1]]

    # --- Filter by intensity percentile ---
    threshold = torch.quantile(intensities, percentile / 100.0)
    mask_percentile = intensities >= threshold

    positions = positions[mask_percentile]
    intensities = intensities[mask_percentile]

    # --- Sort by descending intensity ---
    sorted_indices = torch.argsort(intensities, descending=True)
    positions_sorted = positions[sorted_indices]

    # Return as [x, y] = [col, row]
    return positions_sorted[:, [1, 0]].contiguous()
