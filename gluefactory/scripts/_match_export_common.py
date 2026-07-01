"""Shared scaffolding for the point/line match-export scripts.

`export_point_match_images.py` and `export_line_match_images.py` both visualise
cached eval predictions (``predictions.h5``). This module collects the parts they
share: argument parsing, sample selection, the resize override, IO resolution, and
prediction loading (with friendly errors). The rendering / correctness logic stays
in each script since it differs between points and lines.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from gluefactory.datasets.base_dataset import collate
from gluefactory.eval import get_benchmark
from gluefactory.settings import EVAL_PATH


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def add_common_args(parser: argparse.ArgumentParser, default_benchmark: str):
    """Add the benchmark / selection / resize / IO args shared by both scripts."""
    parser.add_argument("--benchmark", default=default_benchmark)
    parser.add_argument("--exp", required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="Inlier threshold in pixels (homography reprojection / orth-line dist).",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num", type=int, default=None)
    parser.add_argument(
        "--indices",
        default=None,
        help="Comma/range list, e.g. '0,3,10-15'. Overrides --start/--num.",
    )
    parser.add_argument(
        "--random",
        type=int,
        default=None,
        help="Render N random samples (seeded by --seed); overrides --start/--num/"
        "--indices. Use the same --random/--seed as the prediction run so names match.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for --random selection."
    )
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--resize",
        type=int,
        default=None,
        help="Aspect-preserving target edge length (with --side); overrides the "
        "benchmark default resize, e.g. --resize 800 --side long.",
    )
    parser.add_argument(
        "--side",
        default="long",
        choices=["long", "short", "vert", "horz"],
        help="Which image side --resize applies to (aspect preserved).",
    )
    parser.add_argument(
        "--no_resize",
        action="store_true",
        help="Plot at original image resolution (no resize); overrides --resize.",
    )


def parse_indices(args, dataset_size):
    """Sample indices to render: --random (seeded) > --indices > --start/--num."""
    if args.random is not None:
        n = min(args.random, dataset_size)
        rng = np.random.default_rng(args.seed)
        return sorted(rng.choice(dataset_size, size=n, replace=False).tolist())
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


def resolve_io(args, out_subdir):
    """Return (pred_file, out_dir); raise if predictions.h5 is missing."""
    exp_dir = Path(EVAL_PATH) / args.benchmark / args.exp
    pred_file = exp_dir / "predictions.h5"
    if not pred_file.exists():
        raise FileNotFoundError(f"Missing predictions file: {pred_file}")
    out_dir = Path(args.out_dir) if args.out_dir else exp_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    return pred_file, out_dir


def build_loader(args):
    """Benchmark dataloader with an optional aspect-preserving / no-resize override.

    Keys are *set on* the benchmark's preprocessing conf (not replaced wholesale), so
    any other preprocessing options it defines are preserved.
    """
    bench = get_benchmark(args.benchmark)
    data_conf = OmegaConf.create(dict(bench.default_conf["data"]))
    if args.no_resize:
        data_conf.preprocessing.resize = None
    elif args.resize is not None:
        data_conf.preprocessing.side = args.side
        data_conf.preprocessing.resize = args.resize
    return bench.get_dataloader(data_conf)


_COMPAT = {
    "point": (
        "Use a benchmark that exports point matches (keypoints0/1 + matches0):\n"
        "  homography (H_0to1): hpatches, hpatches_extended\n"
        "  relative pose (T_0to1): megadepth1500, megadepth1500_extended, scannet1500\n"
        "Line benchmarks (hpatches_lines, rdnim_lines) export only line matches -- "
        "use export_line_match_images.py for those."
    ),
    "line": (
        "Use a benchmark that exports line matches (homography / H_0to1):\n"
        "  hpatches, hpatches_extended, hpatches_lines, rdnim_lines\n"
        "Pose benchmarks (megadepth1500, scannet1500) export only point matches -- "
        "use export_point_match_images.py for those."
    ),
}


def keys_error(benchmark, missing, kind):
    """Friendly error when a benchmark did not export the needed predictions."""
    return SystemExit(
        f"Benchmark '{benchmark}' exported no {kind} predictions (missing {missing}).\n"
        + _COMPAT[kind]
    )


def load_pred(loader, cache_loader, idx, required_keys, kind, benchmark):
    """Load (data, pred) for one sample with friendly errors.

    Raises a clear SystemExit when the sample is absent from predictions.h5 (usually a
    --random/--seed/--benchmark mismatch with the prediction run) or when the
    benchmark did not export the required keys.
    """
    data = collate([loader.dataset[idx]])
    try:
        pred = cache_loader(data)
    except KeyError:
        name = data["name"][0]
        raise SystemExit(
            f"Sample {name!r} is not in predictions.h5. Make sure --benchmark and "
            "(if used) --random/--seed match the run that produced the predictions."
        ) from None
    missing = [k for k in required_keys if k not in pred]
    if missing:
        raise keys_error(benchmark, missing, kind)
    return data, pred
