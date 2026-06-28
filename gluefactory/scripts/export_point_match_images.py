"""Export side-by-side point-match figures from cached eval predictions.

Usage:
    python -m gluefactory.scripts.export_point_match_images --exp <run_name> \
        [--benchmark hpatches] [--threshold 3.0] [--epi_threshold 5e-4] \
        [--indices 0,3,10-15] [--no_connectors]

How it works:
    - Loads cached per-sample predictions from
      outputs/results/<benchmark>/<exp>/predictions.h5 via CacheLoader
      (reads keypoints0/1 and the matches0 index mapping).
    - Pairs matched keypoints (matches0[i] -> kpts1) into (matched0, matched1).
    - Flags each match correct/incorrect and colors it black/red using whichever
      ground truth the dataset provides:
        * homography H_0to1  -> symmetric reprojection error < threshold (pixels)
        * relative pose T_0to1 + intrinsics -> symmetric epipolar distance
          < epi_threshold (normalized coords, as in eval.utils.eval_matches_epipolar)
      With neither available all matches are drawn black.
    - Renders the pair with plot_images + plot_matches, one PNG per sample.

Compatible benchmarks (must export keypoints0/1 + matches0):
    homography (H_0to1):     hpatches, hpatches_extended
    relative pose (T_0to1):  megadepth1500, megadepth1500_extended, scannet1500
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from gluefactory.datasets.base_dataset import collate
from gluefactory.eval import get_benchmark
from gluefactory.eval.utils import get_matches_scores
from gluefactory.geometry.epipolar import generalized_epi_dist
from gluefactory.geometry.homography import sym_homography_error
from gluefactory.models.cache_loader import CacheLoader
from gluefactory.settings import EVAL_PATH
from gluefactory.visualization.viz2d import plot_images, plot_matches

# Non-interactive backend for headless rendering; safe to switch before any figure.
plt.switch_backend("Agg")

REQUIRED_POINT_KEYS = ["keypoints0", "keypoints1", "matches0"]


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def point_keys_error(benchmark, missing):
    """Friendly error when a benchmark did not export point predictions."""
    return SystemExit(
        f"Benchmark '{benchmark}' exported no point predictions (missing {missing}).\n"
        "Use a benchmark that exports point matches (keypoints0/1 + matches0):\n"
        "  homography (H_0to1): hpatches, hpatches_extended\n"
        "  relative pose (T_0to1): megadepth1500, megadepth1500_extended, scannet1500\n"
        "Line benchmarks (hpatches_lines, rdnim_lines) export only line matches -- "
        "use export_line_match_images.py for those."
    )


def get_point_inliers(kpts0, kpts1, H_0to1, threshold):
    """Homography reprojection inliers (symmetric error in pixels)."""
    errors = sym_homography_error(
        torch.as_tensor(kpts0, dtype=torch.float32),
        torch.as_tensor(kpts1, dtype=torch.float32),
        torch.as_tensor(H_0to1, dtype=torch.float32),
    )
    return errors.numpy() < threshold


def get_point_epipolar_inliers(kpts0, kpts1, camera0, camera1, T_0to1, threshold):
    """Epipolar inliers (symmetric distance in normalized coords, as in eval.utils)."""
    err = generalized_epi_dist(
        torch.as_tensor(kpts0, dtype=torch.float32)[None],
        torch.as_tensor(kpts1, dtype=torch.float32)[None],
        camera0,
        camera1,
        T_0to1,
        all=False,
        essential=True,
    )
    return to_numpy(err[0]) < threshold


def compute_correctness(matched0, matched1, data, args):
    """Flag matches as correct using whichever ground truth the dataset provides.

    Returns (correct_bool_array_or_None, metric_label_or_None): H_0to1 -> "homography"
    reprojection error; T_0to1 + intrinsics -> "epipolar" distance; neither -> (None,
    None) (matches are then drawn without correctness coloring).
    """
    if "H_0to1" in data:
        if len(matched0) == 0:
            return np.zeros(0, dtype=bool), "homography"
        H_0to1 = to_numpy(data["H_0to1"][0])
        correct = get_point_inliers(matched0, matched1, H_0to1, args.threshold)
        return np.asarray(correct, dtype=bool), "homography"
    if "T_0to1" in data and "camera" in data["view0"]:
        if len(matched0) == 0:
            return np.zeros(0, dtype=bool), "epipolar"
        correct = get_point_epipolar_inliers(
            matched0,
            matched1,
            data["view0"]["camera"],
            data["view1"]["camera"],
            data["T_0to1"],
            args.epi_threshold,
        )
        return np.asarray(correct, dtype=bool), "epipolar"
    return None, None


def export_one(loader, cache_loader, idx, out_dir, args):
    data = collate([loader.dataset[idx]])
    pred = cache_loader(data)

    # Guard once, before any rendering: a benchmark with no point predictions
    # (e.g. a line-only benchmark) fails on the first sample with clear guidance.
    missing = [k for k in REQUIRED_POINT_KEYS if k not in pred]
    if missing:
        raise point_keys_error(args.benchmark, missing)

    img0 = data["view0"]["image"][0]
    img1 = data["view1"]["image"][0]
    kpts0 = to_numpy(pred["keypoints0"][0])
    kpts1 = to_numpy(pred["keypoints1"][0])
    m0 = to_numpy(pred["matches0"][0]).astype(int)
    matched0, matched1, _ = get_matches_scores(kpts0, kpts1, m0, np.zeros(len(m0)))

    correct, metric = compute_correctness(matched0, matched1, data, args)

    plot_images([img0, img1], titles=["", ""])
    colors = (
        [[0.0, 0.0, 0.0]] * len(matched0)
        if correct is None
        else np.where(
            correct[:, None],
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[1.0, 0.0, 0.0]]),
        ).tolist()
    )
    plot_matches(
        matched0,
        matched1,
        color=colors,
        lw=0.0 if args.no_connectors else args.line_width,
        ps=args.point_size,
        a=args.alpha,
    )

    n_good = int(correct.sum()) if correct is not None else None
    n_total = len(matched0)

    out = out_dir / f"point_matches_idx{idx:04d}.png"
    plt.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close()
    return out, n_good, n_total, metric


def parse_indices(args, dataset_size):
    if args.indices:
        indices = []
        for chunk in args.indices.split(","):
            if "-" in chunk:
                start, end = map(int, chunk.split("-", 1))
                indices.extend(range(start, end + 1))
            else:
                indices.append(int(chunk))
    else:
        stop = (
            dataset_size
            if args.num is None
            else min(dataset_size, args.start + args.num)
        )
        indices = list(range(args.start, stop))

    return [i for i in indices if 0 <= i < dataset_size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="megadepth1500")
    parser.add_argument("--exp", required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="Homography inlier threshold in pixels (homography benchmarks).",
    )
    parser.add_argument(
        "--epi_threshold",
        type=float,
        default=5e-4,
        help="Epipolar inlier threshold in normalized coords (pose benchmarks).",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num", type=int, default=None)
    parser.add_argument(
        "--indices",
        default=None,
        help="Comma/range list, e.g. '0,3,10-15'. Overrides --start/--num.",
    )
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--line_width", type=float, default=1.0)
    parser.add_argument("--point_size", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--no_connectors", action="store_true")
    args = parser.parse_args()

    exp_dir = Path(EVAL_PATH) / args.benchmark / args.exp
    pred_file = exp_dir / "predictions.h5"
    if not pred_file.exists():
        raise FileNotFoundError(f"Missing predictions file: {pred_file}")

    out_dir = Path(args.out_dir) if args.out_dir else exp_dir / "point_match_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = get_benchmark(args.benchmark).get_dataloader()
    indices = parse_indices(args, len(loader.dataset))
    cache_loader = CacheLoader({"path": str(pred_file), "add_data_path": False})

    if not indices:
        print("No valid indices selected; nothing to export.")
        return

    total_good, total_matches = 0, 0
    metric = None
    last_out = None
    for idx in tqdm(indices, desc="Exporting point match images"):
        last_out, n_good, n_matches, metric_i = export_one(
            loader, cache_loader, idx, out_dir, args
        )
        if metric_i is not None:
            metric = metric_i
            total_good += n_good
        total_matches += n_matches

    print(f"Wrote {len(indices)} images to {out_dir}")
    if metric is None:
        print(
            f"Exported {total_matches} point matches; "
            "no homography/pose ground truth for correctness."
        )
    elif metric == "homography":
        print(
            f"Symmetric homography inliers: {total_good}/{total_matches} "
            f"@ {args.threshold}px"
        )
    else:  # epipolar
        print(
            f"Epipolar inliers: {total_good}/{total_matches} "
            f"@ {args.epi_threshold} (normalized)"
        )
    if last_out is not None:
        print(f"Last image: {last_out}")


if __name__ == "__main__":
    main()
