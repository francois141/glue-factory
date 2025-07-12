import csv
from pathlib import Path

import cv2
import kornia
import numpy as np
import torch


def read_image(path, grayscale=False):
    """Read an image from path as RGB or grayscale"""
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise IOError(f"Could not read image at {path}.")
    if not grayscale:
        image = image[..., ::-1]
    return image


def numpy_image_to_torch(image):
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)


def rotate_intrinsics(K, image_shape, rot):
    """image_shape is the shape of the image after rotation"""
    assert rot <= 3
    h, w = image_shape[:2][:: -1 if (rot % 2) else 1]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    rot = rot % 4
    if rot == 1:
        return np.array(
            [[fy, 0.0, cy], [0.0, fx, w - cx], [0.0, 0.0, 1.0]], dtype=K.dtype
        )
    elif rot == 2:
        return np.array(
            [[fx, 0.0, w - cx], [0.0, fy, h - cy], [0.0, 0.0, 1.0]],
            dtype=K.dtype,
        )
    else:  # if rot == 3:
        return np.array(
            [[fy, 0.0, h - cy], [0.0, fx, cx], [0.0, 0.0, 1.0]], dtype=K.dtype
        )


def rotate_pose_inplane(i_T_w, rot):
    rotation_matrices = [
        np.array(
            [
                [np.cos(r), -np.sin(r), 0.0, 0.0],
                [np.sin(r), np.cos(r), 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        for r in [np.deg2rad(d) for d in (0, 270, 180, 90)]
    ]
    return np.dot(rotation_matrices[rot], i_T_w)


def scale_intrinsics(K, scales):
    """Scale intrinsics after resizing the corresponding image."""
    scales = np.diag(np.concatenate([scales, [1.0]]))
    return np.dot(scales.astype(K.dtype, copy=False), K)


def get_divisible_wh(w, h, df=None):
    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w, h])
    else:
        w_new, h_new = w, h
    return w_new, h_new


def resize(image, size, fn=None, interp="linear", df=None):
    """Resize an image to a fixed size, or according to max or min edge."""
    h, w = image.shape[:2]
    if isinstance(size, int):
        scale = size / fn(h, w)
        h_new, w_new = int(round(h * scale)), int(round(w * scale))
        w_new, h_new = get_divisible_wh(w_new, h_new, df)
        scale = (w_new / w, h_new / h)
    elif isinstance(size, (tuple, list)):
        h_new, w_new = size
        scale = (w_new / w, h_new / h)
    else:
        raise ValueError(f"Incorrect new size: {size}")
    mode = {
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "nearest": cv2.INTER_NEAREST,
        "area": cv2.INTER_AREA,
    }[interp]
    return cv2.resize(image, (w_new, h_new), interpolation=mode), scale


def resize_img_kornia(img: torch.Tensor, size: int) -> torch.Tensor:
    """
    This resize function has similar functionality to ImagePreprocessor resize.
    Here we resize and keep aspect ratio by scaling long side to 'size' abd scale the other part accoringly.

    Args:
        img (torch.Tensor): image to resize
        size (int): shape to resize to

    Returns:
        torch.Tensor: reshaped image
    """
    resized = kornia.geometry.transform.resize(
        img,
        size,
        side="long",
        antialias=True,
        align_corners=None,
        interpolation="bilinear",
    )
    return resized


def crop(image, size, random=True, other=None, K=None, return_bbox=False):
    """Random or deterministic crop of an image, adjust depth and intrinsics."""
    h, w = image.shape[:2]
    h_new, w_new = (size, size) if isinstance(size, int) else size
    top = np.random.randint(0, h - h_new + 1) if random else 0
    left = np.random.randint(0, w - w_new + 1) if random else 0
    image = image[top : top + h_new, left : left + w_new]
    ret = [image]
    if other is not None:
        ret += [other[top : top + h_new, left : left + w_new]]
    if K is not None:
        K[0, 2] -= left
        K[1, 2] -= top
        ret += [K]
    if return_bbox:
        ret += [(top, top + h_new, left, left + w_new)]
    return ret


def zero_pad(size, *images):
    """zero pad images to size x size"""
    ret = []
    for image in images:
        if image is None:
            ret.append(None)
            continue
        h, w = image.shape[:2]
        padded = np.zeros((size, size) + image.shape[2:], dtype=image.dtype)
        padded[:h, :w] = image
        ret.append(padded)
    return ret

def warp_points(points, H):
    """Warp 2D points by an homography H."""
    n_points = points.shape[0]
    reproj_points = points.copy()[:, [1, 0]]
    reproj_points = np.concatenate([reproj_points, np.ones((n_points, 1))], axis=1)
    reproj_points = H.dot(reproj_points.transpose()).transpose()
    reproj_points = reproj_points[:, :2] / reproj_points[:, 2:]
    reproj_points = reproj_points[:, [1, 0]]
    return reproj_points


def warp_lines(lines, H):
    """Warp lines of the shape [N, 2, 2] by an homography H."""
    return warp_points(lines.reshape(-1, 2), H).reshape(-1, 2, 2)


def read_timestamps(text_file: Path) -> dict[str, list[float]]:
    """
    Read a text file containing the timestamps of images
    and return a dictionary matching the name of the image
    to its timestamp.
    """
    timestamps = {"name": [], "date": [], "hour": [], "minute": [], "time": []}
    with open(text_file, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=" ")
        for row in reader:
            timestamps["name"].append(row[0])
            timestamps["date"].append(row[1])
            hour = int(row[2])
            timestamps["hour"].append(hour)
            minute = int(row[3])
            timestamps["minute"].append(minute)
            timestamps["time"].append(hour + minute / 60.0)
    return timestamps
