"""Export side-by-side line-match figures from cached eval predictions.

Usage:
    python -m gluefactory.scripts.export_line_match_images --exp <run_name> \
        [--benchmark hpatches] [--threshold 3.0] [--indices 0,3,10-15] [--no_connectors]

How it works:
    - Loads cached per-sample predictions from
      outputs/results/<benchmark>/<exp>/predictions.h5 via CacheLoader
      (reads lines0/1 and the line_matches0/1 index mapping).
    - Pairs matched segments (line_matches0[i] -> lines1) into (matched0, matched1).
    - If the dataset has a ground-truth homography H_0to1, labels each match by
      orthogonal line distance after warping < threshold and colors correct black /
      wrong red; without H_0to1 all matches are drawn black.
    - Saves one PNG per pair via plot_images + plot_red_black_line_matches.

Compatible benchmarks (must export lines0/1 + line_matches0/1):
    homography (H_0to1): hpatches, hpatches_extended, hpatches_lines, rdnim_lines
Line correctness needs a homography, so pose benchmarks (megadepth1500, scannet1500)
are unsupported -- they export only point matches; the script exits with guidance.
"""

import argparse
from pathlib import Path

import matplotlib.lines
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from gluefactory.datasets.base_dataset import collate
from gluefactory.datasets.homographies_deeplsd import warp_lines
from gluefactory.eval import get_benchmark
from gluefactory.models.cache_loader import CacheLoader
from gluefactory.models.lines.line_distances import get_orth_line_dist_torch
from gluefactory.settings import EVAL_PATH
from gluefactory.visualization.viz2d import plot_images

# Non-interactive backend for headless rendering; safe to switch before any figure.
plt.switch_backend("Agg")

REQUIRED_LINE_KEYS = ["lines0", "lines1", "line_matches0", "line_matches1"]


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def line_keys_error(benchmark, missing):
    """Friendly error when a benchmark did not export line predictions."""
    return SystemExit(
        f"Benchmark '{benchmark}' exported no line predictions (missing {missing}).\n"
        "Use a benchmark that exports line matches (homography / H_0to1):\n"
        "  hpatches, hpatches_extended, hpatches_lines, rdnim_lines\n"
        "Pose benchmarks (megadepth1500, scannet1500) export only point matches -- "
        "use export_point_match_images.py for those."
    )


def get_pair_indices(lines0, lines1, pred):
    """Matched index pairs from line matches.

    Line matchers emit line_matches0/line_matches1 as *paired* index arrays (equal
    length == number of matches), unlike point matches0 which is a per-line mapping;
    we support both (mapping detected by len(line_matches0) == len(lines0)).
    """
    m0 = to_numpy(pred["line_matches0"][0]).astype(int)
    m1 = to_numpy(pred["line_matches1"][0]).astype(int)

    if len(m0) == len(lines0):
        idx0 = np.where(m0 > -1)[0]
        idx1 = m0[idx0]
    else:
        idx0, idx1 = m0, m1

    valid = (idx0 >= 0) & (idx0 < len(lines0)) & (idx1 >= 0) & (idx1 < len(lines1))
    return idx0[valid], idx1[valid]


def get_line_inliers_and_error(lines0, lines1, H_0to1, threshold):
    warped_lines1 = warp_lines(lines1, np.linalg.inv(H_0to1))
    dist = get_orth_line_dist_torch(
        torch.as_tensor(lines0, dtype=torch.float32),
        torch.as_tensor(warped_lines1, dtype=torch.float32),
    )
    dist = torch.diag(dist).detach().cpu().numpy()
    inliers = dist < threshold
    return inliers


def should_flip_lines(lines, image, convention):
    """Return whether line coordinates must be converted from row/col to x/y."""
    if convention == "xy":
        return False
    if convention == "ij":
        return True
    if len(lines) == 0:
        return False

    _, height, width = image.shape
    max_first = np.nanmax(lines[..., 0])
    max_second = np.nanmax(lines[..., 1])
    looks_ij = max_first <= height * 1.05 and max_second <= width * 1.05
    looks_xy = max_first <= width * 1.05 and max_second <= height * 1.05
    if looks_ij and not looks_xy:
        return True
    return False


def lines_to_plot_coords(lines, image, convention):
    """Convert lines to Matplotlib coordinates when needed."""
    if should_flip_lines(lines, image, convention):
        return lines[..., [1, 0]]
    return lines


def plot_red_black_line_matches(
    lines0,
    lines1,
    correct,
    lw=1.3,
    alpha=0.85,
    endpoint_size=2.0,
    draw_connectors=True,
):
    fig = plt.gcf()
    axes = fig.axes[:2]
    if correct is None:
        colors = np.tile(np.array([[0.0, 0.0, 0.0]]), (len(lines0), 1))
    else:
        colors = np.where(
            correct[:, None],
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[1.0, 0.0, 0.0]]),
        )

    for line0, line1, color in zip(lines0, lines1, colors):
        for ax, line in zip(axes, [line0, line1]):
            ax.add_line(
                matplotlib.lines.Line2D(
                    (line[0, 0], line[1, 0]),
                    (line[0, 1], line[1, 1]),
                    c=color,
                    linewidth=lw,
                    alpha=alpha,
                    zorder=2,
                )
            )
            if endpoint_size > 0:
                ax.scatter(
                    line[:, 0],
                    line[:, 1],
                    c=[color],
                    s=endpoint_size,
                    linewidths=0,
                    alpha=alpha,
                    zorder=3,
                )

        if draw_connectors:
            c0 = line0.mean(axis=0)
            c1 = line1.mean(axis=0)
            connector = matplotlib.patches.ConnectionPatch(
                xyA=tuple(c0),
                xyB=tuple(c1),
                coordsA=axes[0].transData,
                coordsB=axes[1].transData,
                axesA=axes[0],
                axesB=axes[1],
                color=color,
                linewidth=max(0.3, lw * 0.5),
                alpha=min(alpha, 0.35),
                zorder=1,
                clip_on=True,
            )
            connector.set_annotation_clip(True)
            fig.add_artist(connector)


def export_one(loader, cache_loader, idx, threshold, out_dir, args):
    data = collate([loader.dataset[idx]])
    pred = cache_loader(data)

    # Guard once, before any rendering: a benchmark with no line predictions
    # (e.g. a pose benchmark) fails on the first sample with clear guidance.
    missing = [k for k in REQUIRED_LINE_KEYS if k not in pred]
    if missing:
        raise line_keys_error(args.benchmark, missing)

    img0 = data["view0"]["image"][0]
    img1 = data["view1"]["image"][0]
    lines0 = to_numpy(pred["lines0"][0])
    lines1 = to_numpy(pred["lines1"][0])
    idx0, idx1 = get_pair_indices(lines0, lines1, pred)

    matched0 = lines0[idx0]
    matched1 = lines1[idx1]

    if len(matched0) == 0:
        correct = np.zeros(0, dtype=bool) if "H_0to1" in data else None
    elif "H_0to1" in data:
        H_0to1 = to_numpy(data["H_0to1"][0])
        correct = get_line_inliers_and_error(matched0, matched1, H_0to1, threshold)
        correct = np.asarray(correct, dtype=bool)
    else:
        correct = None

    plot_images([img0, img1], titles=["", ""])
    plot_lines0 = lines_to_plot_coords(matched0, img0, args.line_convention)
    plot_lines1 = lines_to_plot_coords(matched1, img1, args.line_convention)

    plot_red_black_line_matches(
        plot_lines0,
        plot_lines1,
        correct,
        lw=args.line_width,
        alpha=args.alpha,
        endpoint_size=args.endpoint_size,
        draw_connectors=not args.no_connectors,
    )

    n_good = int(correct.sum()) if correct is not None else None
    n_total = len(matched0)

    out = out_dir / f"line_matches_idx{idx:04d}.png"
    plt.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close()
    return out, n_good, n_total


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
    parser.add_argument("--benchmark", default="hpatches")
    parser.add_argument("--exp", required=True)
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num", type=int, default=None)
    parser.add_argument(
        "--indices",
        default=None,
        help="Comma/range list, e.g. '0,3,10-15'. Overrides --start/--num.",
    )
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--line_convention",
        choices=["auto", "xy", "ij"],
        default="auto",
        help="Use ij for row/col lines, xy for Matplotlib-ready lines.",
    )
    parser.add_argument("--line_width", type=float, default=2.0)
    parser.add_argument("--endpoint_size", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--no_connectors", action="store_true")
    args = parser.parse_args()

    exp_dir = Path(EVAL_PATH) / args.benchmark / args.exp
    pred_file = exp_dir / "predictions.h5"
    if not pred_file.exists():
        raise FileNotFoundError(f"Missing predictions file: {pred_file}")

    out_dir = Path(args.out_dir) if args.out_dir else exp_dir / "line_match_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = get_benchmark(args.benchmark).get_dataloader()
    indices = parse_indices(args, len(loader.dataset))
    cache_loader = CacheLoader({"path": str(pred_file), "add_data_path": False})

    if not indices:
        print("No valid indices selected; nothing to export.")
        return

    total_good, total_matches = 0, 0
    has_correctness = True
    last_out = None
    for idx in tqdm(indices, desc="Exporting line match images"):
        last_out, n_good, n_matches = export_one(
            loader, cache_loader, idx, args.threshold, out_dir, args
        )
        if n_good is None:
            has_correctness = False
        else:
            total_good += n_good
        total_matches += n_matches

    print(f"Wrote {len(indices)} images to {out_dir}")
    if not has_correctness:
        print(
            f"Exported {total_matches} line matches; "
            "no homography correctness available."
        )
    else:
        print(
            f"Orthogonal-distance inliers: {total_good}/{total_matches} "
            f"@ {args.threshold}px"
        )
    if last_out is not None:
        print(f"Last image: {last_out}")


if __name__ == "__main__":
    main()
