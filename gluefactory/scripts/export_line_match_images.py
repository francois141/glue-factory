"""Export side-by-side line-match figures from cached eval predictions.

Reads a benchmark's cached predictions.h5 (written by gluefactory.eval.*) and draws,
per image pair, the matched line segments colored by correctness against the
ground-truth homography. Outputs one PNG per sample under
outputs/results/<benchmark>/<exp>/line_match_images/ (or --out_dir).

Usage:
    python -m gluefactory.scripts.export_line_match_images --exp <run> [options]

    # first 100 hpatches_extended pairs, aspect-preserving 800px long side:
    ... --benchmark hpatches_extended --exp <run> --num 100 --side long --resize 800
    # 100 reproducible random samples (SAME seed as the prediction run):
    ... --exp <run> --random 100 --seed 0
    # declutter: only the 300 longest segments, at native resolution:
    ... --exp <run> --no_resize --max_matches 300
    # repo-style multicolor view instead of green/red:
    ... --exp <run> --color

Selecting samples (predictions.h5 is keyed by sample name):
    --indices 0,3,10-15    explicit indices / ranges
    --start / --num        a contiguous slice (default: all samples)
    --random N --seed S    N seeded-random indices. IMPORTANT: use the SAME
                           --random/--seed/--benchmark as the prediction run, else the
                           sample names are not found.

Image size: (default) benchmark resize | --side long --resize 800 | --no_resize.

Coloring (vs ground-truth homography H): green = correct, red = wrong, black = no H.
    "correct" = orthogonal line distance to the H-warped segment < --threshold (px).
    --color                multicolor view (plot_color_line_matches): a distinct color
                           per match, wrong matches dimmed (no green/red).
    --max_matches N        draw only the top-N matches (declutter); --line_rank picks
                           the ranking: 'length' (longest, default) or 'score'
                           (line_matching_scores0). Summary then counts only those N.

Compatible benchmarks (must export lines0/1 + line_matches0/1):
    homography (H_0to1): hpatches, hpatches_extended, hpatches_lines, rdnim_lines
    (pose benchmarks export only point matches; the script exits with guidance.)
"""

import argparse

import matplotlib.lines
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from gluefactory.datasets.homographies_deeplsd import warp_lines
from gluefactory.models.cache_loader import CacheLoader
from gluefactory.models.lines.line_distances import get_orth_line_dist_torch
from gluefactory.scripts._match_export_common import (
    add_common_args,
    build_loader,
    load_pred,
    parse_indices,
    resolve_io,
    to_numpy,
)
from gluefactory.visualization.viz2d import plot_color_line_matches, plot_images

# Non-interactive backend for headless rendering; safe to switch before any figure.
plt.switch_backend("Agg")

REQUIRED_LINE_KEYS = ["lines0", "lines1", "line_matches0", "line_matches1"]


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


def top_line_matches(idx0, idx1, lines0, lines1, pred, n, rank):
    """Keep the top-n matched line pairs.

    rank='length' (default): the longest segments (mean endpoint distance of the two
    matched lines). rank='score': line_matching_scores0 descriptor similarity.
    """
    if rank == "length":
        len0 = np.linalg.norm(lines0[idx0][:, 0] - lines0[idx0][:, 1], axis=1)
        len1 = np.linalg.norm(lines1[idx1][:, 0] - lines1[idx1][:, 1], axis=1)
        key = (len0 + len1) / 2.0
    else:  # "score"
        sc = (
            to_numpy(pred["line_matching_scores0"][0])
            if "line_matching_scores0" in pred
            else None
        )
        if sc is not None and len(sc) == len(lines0):
            key = sc[idx0]  # mapping-style scores (one per line0)
        elif sc is not None and len(sc) == len(idx0):
            key = sc  # per-match scores
        else:
            key = None
    order = np.argsort(key)[::-1][:n] if key is not None else np.arange(n)
    return idx0[order], idx1[order]


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


def plot_correctness_line_matches(
    lines0,
    lines1,
    correct,
    lw=1.3,
    alpha=0.85,
    endpoint_size=2.0,
    draw_connectors=True,
):
    """Draw matched segments colored by correctness: green=correct, red=wrong.

    When `correct` is None (no ground-truth homography) all matches are drawn black.
    """
    fig = plt.gcf()
    axes = fig.axes[:2]
    if correct is None:
        colors = np.tile(np.array([[0.0, 0.0, 0.0]]), (len(lines0), 1))
    else:
        colors = np.where(
            correct[:, None],
            np.array([[0.0, 1.0, 0.0]]),
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


def export_one(loader, cache_loader, idx, out_dir, args):
    data, pred = load_pred(
        loader, cache_loader, idx, REQUIRED_LINE_KEYS, "line", args.benchmark
    )

    img0 = data["view0"]["image"][0]
    img1 = data["view1"]["image"][0]
    lines0 = to_numpy(pred["lines0"][0])
    lines1 = to_numpy(pred["lines1"][0])
    idx0, idx1 = get_pair_indices(lines0, lines1, pred)
    if args.max_matches is not None and len(idx0) > args.max_matches:
        idx0, idx1 = top_line_matches(
            idx0, idx1, lines0, lines1, pred, args.max_matches, args.line_rank
        )

    matched0 = lines0[idx0]
    matched1 = lines1[idx1]

    if len(matched0) == 0:
        correct = np.zeros(0, dtype=bool) if "H_0to1" in data else None
    elif "H_0to1" in data:
        H_0to1 = to_numpy(data["H_0to1"][0])
        correct = get_line_inliers_and_error(matched0, matched1, H_0to1, args.threshold)
        correct = np.asarray(correct, dtype=bool)
    else:
        correct = None

    plot_images([img0, img1], titles=["", ""])
    plot_lines0 = lines_to_plot_coords(matched0, img0, args.line_convention)
    plot_lines1 = lines_to_plot_coords(matched1, img1, args.line_convention)

    if args.color:
        # Repo-style multicolor: each match a distinct color across both images;
        # wrong matches (when correctness is known) are dimmed to low alpha.
        plot_color_line_matches([plot_lines0, plot_lines1], correct, lw=args.line_width)
    else:
        # Binary correctness: green=correct, red=wrong (black when no GT homography).
        plot_correctness_line_matches(
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


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_benchmark="hpatches")
    parser.add_argument(
        "--max_matches",
        type=int,
        default=None,
        help="Draw only the top-N matches (declutter); ranked by --line_rank. The "
        "inlier summary then counts only the kept N.",
    )
    parser.add_argument(
        "--line_rank",
        choices=["length", "score"],
        default="length",
        help="Ranking for --max_matches: 'length' (longest segments, default) or "
        "'score' (line_matching_scores0 descriptor similarity).",
    )
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
    parser.add_argument(
        "--color",
        action="store_true",
        help="Render repo-style multicolor matches (plot_color_line_matches) instead "
        "of green/red correctness coloring; wrong matches are dimmed.",
    )
    args = parser.parse_args()

    pred_file, out_dir = resolve_io(args, "line_match_images")
    loader = build_loader(args)
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
            loader, cache_loader, idx, out_dir, args
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
