from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from gluefactory.datasets import get_dataset
from gluefactory.eval.eval_pipeline import EvalPipeline
from gluefactory.eval.io import get_eval_parser, load_model, parse_eval_args
from gluefactory.models.cache_loader import CacheLoader
from gluefactory.models.utils.metrics_lines import (
    compute_loc_error,
    compute_repeatability,
)
from gluefactory.settings import EVAL_PATH
from gluefactory.utils.export_predictions import export_predictions
from gluefactory.utils.tensor import map_tensor

from gluefactory.datasets.homographies_deeplsd import warp_lines
from gluefactory.models.lines.line_utils import H_estimation


class RDNIMPipeline(EvalPipeline):
    default_conf = {
        "data": {
            "batch_size": 1,
            "name": "rdnim",
            "num_workers": 2,
            "preprocessing": {
                "resize": 480,  # we also resize during eval to have comparable metrics
                "side": "short",
            },
        },
        "model": {
            "ground_truth": {
                "name": None,  # remove gt matches
            }
        },
        "eval": {
            "estimator": "poselib",
            "ransac_th": 1.0,  # -1 runs a bunch of thresholds and selects the best
        },
        "use_points": False,
        "use_lines": True,
        "repeatability_th": [1, 3, 5],
        "num_lines_th": [10, 50, 300],
    }
    export_keys = []

    optional_export_keys = [
        "lines0",
        "lines1",
        "orig_lines0",
        "orig_lines1",
        "line_matches0",
        "line_matches1",
        "line_matching_scores0",
        "line_matching_scores1",
        "line_distances",
    ]

    def _init(self, conf):
        if conf.use_points:
            self.export_keys += [
                "keypoints0",
                "keypoints1",
                "keypoint_scores0",
                "keypoint_scores1",
                "matches0",
                "matches1",
                "matching_scores0",
                "matching_scores1",
            ]
        if conf.use_lines:
            self.export_keys += [
                "lines0",
                "lines1",
                "line_matches0",
                "line_matches1",
                "line_matching_scores0",
                "line_matching_scores1",
            ]

    @classmethod
    def get_dataloader(self, data_conf=None):
        data_conf = data_conf if data_conf else self.default_conf["data"]
        dataset = get_dataset("rdnim")(data_conf)
        return dataset.get_data_loader("test")

    def get_predictions(self, experiment_dir, model=None, overwrite=False):
        pred_file = experiment_dir / "predictions.h5"
        if not pred_file.exists() or overwrite:
            if model is None:
                model = load_model(self.conf.model, self.conf.checkpoint)
            export_predictions(
                self.get_dataloader(self.conf.data),
                model,
                pred_file,
                keys=self.export_keys,
                optional_keys=self.optional_export_keys,
            )
        return pred_file

    def run_eval(self, loader: torch.utils.data.DataLoader, pred_file: Path, plot: bool):
        assert pred_file.exists()
        results = defaultdict(list)
        cache_loader = CacheLoader({"path": str(pred_file), "collate": None}).eval()

        def evaluate_sample(i, data):
            pred = cache_loader(data)
            data = map_tensor(data, lambda t: torch.squeeze(t, dim=0))
            results_i = {
                "names": data["name"][0],
                "scenes": data["scene"][0],
            }

            # Compute H_err
            segs1, segs2 = pred["lines0"], pred["lines1"]
            matched_idx1 = pred["line_matches0"].to(torch.int64)
            matched_idx2 = pred["line_matches1"].to(torch.int64)

            H = data["H_0to1"].cpu().numpy()

            for thresh in [1, 3, 5]:
                if len(matched_idx1) < 3:
                    results_i[f"H_err@{thresh}"] = 0
                else:
                    matched_seg1 = segs1[matched_idx1].cpu().numpy()
                    matched_seg2 = warp_lines(segs2.cpu().numpy(), H)[matched_idx2]
                    results_i[f"H_err@{thresh}"] = H_estimation(
                        matched_seg1,
                        matched_seg2,
                        H,
                        data["view0"]["image"].shape[1:],
                        reproj_thresh=thresh,
                    )[0]

            # Repeatability and localization error
            if "lines0" in pred:
                lines0 = pred["lines0"].cpu()
                lines1 = pred["lines1"].cpu()

                if plot:
                    plot_images(
                        [
                            data["view0"]["image"].permute(1, 2, 0),
                            data["view1"]["image"].permute(1, 2, 0),
                        ],
                        ["H0", "H1"],
                    )
                    plot_lines(lines=[pred["orig_lines0"], pred["orig_lines1"]])
                    save_plot(os.path.join("./match_score/", f"{i}.jpg"))
                    plt.close()

                results_i["repeatability"] = compute_repeatability(
                    lines0,
                    lines1,
                    pred["line_matches0"].cpu(),
                    pred["line_matches1"].cpu(),
                    pred["line_matching_scores0"].cpu(),
                    self.conf.repeatability_th,
                    rep_type="num",
                )
                results_i["loc_error"] = compute_loc_error(
                    pred["line_matching_scores0"].cpu(), self.conf.num_lines_th
                )
                results_i["num_lines"] = (lines0.shape[0] + lines1.shape[0]) / 2

            return results_i

        # Run in parallel with a progress bar
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(evaluate_sample, i, data): i
                for i, data in enumerate(loader)
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
                results_i = future.result()
                for k, v in results_i.items():
                    results[k].append(v)

        # Summarize results
        summaries = {}
        for k, v in results.items():
            arr = np.array(v)
            if not np.issubdtype(arr.dtype, np.number):
                continue
            if k.startswith("H_err"):
                summaries[f"m{k}"] = round(np.mean(arr), 3)
            else:
                summaries[f"m{k}"] = round(np.median(arr), 3)

        if "repeatability" in results:
            for i, th in enumerate(self.conf.repeatability_th):
                values = [x[i] for x in results["repeatability"]]
                summaries[f"repeatability@{th}px"] = round(np.median(values), 3)

        if "loc_error" in results:
            for i, th in enumerate(self.conf.num_lines_th):
                values = [x[i] for x in results["loc_error"]]
                summaries[f"loc_error@{th}lines"] = round(np.median(values), 3)

        figures = {}
        return summaries, figures, results


if __name__ == "__main__":
    dataset_name = Path(__file__).stem
    parser = get_eval_parser()
    args = parser.parse_intermixed_args()

    default_conf = OmegaConf.create(RDNIMPipeline.default_conf)

    # mingle paths
    output_dir = Path(EVAL_PATH, dataset_name)
    output_dir.mkdir(exist_ok=True, parents=True)

    name, conf = parse_eval_args(
        dataset_name,
        args,
        "configs/",
        default_conf,
    )

    experiment_dir = output_dir / name
    experiment_dir.mkdir(exist_ok=True)

    pipeline = RDNIMPipeline(conf)
    s, f, r = pipeline.run(
        experiment_dir, overwrite=args.overwrite, overwrite_eval=args.overwrite_eval
    )

    # print results
    pprint(s)
    if args.plot:
        for name, fig in f.items():
            fig.canvas.manager.set_window_title(name)
        plt.show()
