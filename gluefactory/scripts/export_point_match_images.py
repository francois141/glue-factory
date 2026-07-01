"""Export side-by-side point-match figures from cached eval predictions.

Reads a benchmark's cached predictions.h5 (written by gluefactory.eval.*) and draws,
per image pair, the matched keypoints colored by correctness against the ground truth.
Outputs one PNG per sample under outputs/results/<benchmark>/<exp>/point_match_images/
(or --out_dir).

Usage:
    python -m gluefactory.scripts.export_point_match_images --exp <run> [options]

    # first 100 hpatches_extended pairs, aspect-preserving 800px long side:
    ... --benchmark hpatches_extended --exp <run> --num 100 --side long --resize 800
    # 100 reproducible random samples (SAME seed as the prediction run):
    ... --exp <run> --random 100 --seed 0
    # declutter: only the 300 most-salient matches, at native resolution:
    ... --exp <run> --no_resize --max_matches 300

Selecting samples (predictions.h5 is keyed by sample name):
    --indices 0,3,10-15    explicit indices / ranges
    --start / --num        a contiguous slice (default: all samples)
    --random N --seed S    N seeded-random indices. IMPORTANT: use the SAME
                           --random/--seed/--benchmark as the run that produced the
                           predictions, else the sample names are not found.

Image size (display + coord scaling; predictions store original-image coords):
    (default)              the benchmark's own resize
    --side long --resize 800   aspect-preserving, long side = 800
    --no_resize            original resolution

Coloring (vs ground truth): yellow = correct, red = wrong, black = no ground truth.
    * homography H_0to1  -> symmetric reprojection error < --threshold (px)
    * relative pose T_0to1 + intrinsics -> epipolar distance < --epi_threshold
    --max_matches N        draw only the top-N matches by keypoint detection score
                           (declutter); the inlier summary then counts only those N.

Compatible benchmarks (must export keypoints0/1 + matches0):
    homography (H_0to1):    hpatches, hpatches_extended
    relative pose (T_0to1): megadepth1500, megadepth1500_extended, scannet1500
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from gluefactory.eval.utils import get_matches_scores
from gluefactory.geometry.epipolar import generalized_epi_dist
from gluefactory.geometry.homography import sym_homography_error
from gluefactory.models.cache_loader import CacheLoader
from gluefactory.scripts._match_export_common import (
    add_common_args,
    build_loader,
    load_pred,
    parse_indices,
    resolve_io,
    to_numpy,
)
from gluefactory.visualization.viz2d import plot_images, plot_matches

# Non-interactive backend for headless rendering; safe to switch before any figure.
plt.switch_backend("Agg")

REQUIRED_POINT_KEYS = ["keypoints0", "keypoints1", "matches0"]


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
    data, pred = load_pred(
        loader, cache_loader, idx, REQUIRED_POINT_KEYS, "point", args.benchmark
    )

    img0 = data["view0"]["image"][0]
    img1 = data["view1"]["image"][0]
    kpts0 = to_numpy(pred["keypoints0"][0])
    kpts1 = to_numpy(pred["keypoints1"][0])
    m0 = to_numpy(pred["matches0"][0]).astype(int)
    matched0, matched1, _ = get_matches_scores(kpts0, kpts1, m0, np.zeros(len(m0)))
    if args.max_matches is not None and len(matched0) > args.max_matches:
        # nn_point_line's matching_scores0 is binary (matched/not), so rank by the
        # keypoint detection score -- the only graded point confidence available.
        ks0 = (
            to_numpy(pred["keypoint_scores0"][0])[m0 > -1]
            if "keypoint_scores0" in pred
            else np.zeros(len(matched0))
        )
        keep = np.argsort(ks0)[::-1][: args.max_matches]
        matched0, matched1 = matched0[keep], matched1[keep]

    correct, metric = compute_correctness(matched0, matched1, data, args)

    plot_images([img0, img1], titles=["", ""])
    colors = (
        [[0.0, 0.0, 0.0]] * len(matched0)  # no ground truth -> neutral black
        if correct is None
        else np.where(
            correct[:, None],
            np.array([[1.0, 1.0, 0.0]]),  # correct -> yellow
            np.array([[1.0, 0.0, 0.0]]),  # wrong -> red
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


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_benchmark="megadepth1500")
    parser.add_argument(
        "--epi_threshold",
        type=float,
        default=5e-4,
        help="Epipolar inlier threshold in normalized coords (pose benchmarks).",
    )
    parser.add_argument(
        "--max_matches",
        type=int,
        default=None,
        help="Draw only the top-N matches, ranked by keypoint detection score "
        "(declutter; nn_point_line has no graded point-match score). The inlier "
        "summary then counts only the kept N.",
    )
    parser.add_argument("--line_width", type=float, default=1.0)
    parser.add_argument("--point_size", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--no_connectors", action="store_true")
    args = parser.parse_args()

    pred_file, out_dir = resolve_io(args, "point_match_images")
    loader = build_loader(args)
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
