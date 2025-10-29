import copy
import itertools
import os
from collections import defaultdict
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils
import torch.utils.data
from omegaconf import OmegaConf
from tqdm import tqdm

from gluefactory.datasets import get_dataset
from gluefactory.datasets.homographies_deeplsd import warp_lines
from gluefactory.eval.eval_pipeline import (
    EvalPipeline,
    exists_eval,
    load_eval,
    save_eval,
)
from gluefactory.eval.io import get_eval_parser, load_model, parse_eval_args
from gluefactory.models import BaseModel
from gluefactory.models.cache_loader import CacheLoader
from gluefactory.models.lines.line_utils import H_estimation
from gluefactory.models.utils.metrics_lines import (
    compute_loc_error,
    compute_repeatability,
)
from gluefactory.settings import EVAL_PATH
from gluefactory.utils.export_predictions import export_predictions
from gluefactory.utils.tensor import map_tensor
from gluefactory.visualization.viz2d import plot_images, plot_lines, save_plot


class HPatchesPipeline(EvalPipeline):
    default_conf = {
        "data": {
            "batch_size": 1,
            "name": "hpatches",
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
        dataset = get_dataset("hpatches")(data_conf)
        return dataset.get_data_loader("test")

    def get_predictions(
        self,
        experiment_dir: Path,
        model: BaseModel | None = None,
        overwrite: bool = False,
    ) -> Path:
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

    def run(
        self,
        experiment_dir: Path,
        model: BaseModel | None = None,
        overwrite=False,
        overwrite_eval=False,
        plot=False,
    ):
        """Run export+eval loop"""
        self.save_conf(
            experiment_dir, overwrite=overwrite, overwrite_eval=overwrite_eval
        )
        pred_file = self.get_predictions(
            experiment_dir, model=model, overwrite=overwrite
        )

        f = {}
        if not exists_eval(experiment_dir) or overwrite_eval or overwrite:
            s, f, r = self.run_eval(self.get_dataloader(), pred_file, plot)
            save_eval(experiment_dir, s, f, r)
        s, r = load_eval(experiment_dir)
        return s, f, r

    def run_eval(
        self, loader: torch.utils.data.DataLoader, pred_file: Path, plot: bool
    ):
        assert pred_file.exists()
        results = defaultdict(list)

        cache_loader = CacheLoader({"path": str(pred_file), "collate": None}).eval()
        for i, data in enumerate(tqdm(loader)):
            # if i in range(360,365):
            #     continue
            pred = cache_loader(data)
            # Remove batch dimension
            data = map_tensor(data, lambda t: torch.squeeze(t, dim=0))
            # add custom evaluations here

            results_i = {}

            # we also store the names for later reference
            results_i["names"] = data["name"][0]
            results_i["scenes"] = data["scene"][0]

            # compute H_err
            segs1, segs2, matched_idx1, matched_idx2 = (
                pred["lines0"],
                pred["lines1"],
                pred["line_matches0"].to(torch.int64),
                pred["line_matches1"].to(torch.int64),
            )

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

            # compute repeatability and loc_error
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

            for k, v in results_i.items():
                results[k].append(v)

        # summarize results as a dict[str, float]
        # you can also add your custom evaluations here
        summaries = {}
        for k, v in results.items():
            arr = np.array(v)
            if not np.issubdtype(np.array(v).dtype, np.number):
                continue
            if k.startswith("H_err"):
                summaries[f"m{k}"] = round(np.mean(arr), 3)
            else:
                summaries[f"m{k}"] = round(np.median(arr), 3)

        if "repeatability" in results.keys():
            for i, th in enumerate(self.conf.repeatability_th):
                cur_nums = list(map(lambda x: x[i], results["repeatability"]))
                summaries[f"repeatability@{th}px"] = round(np.median(cur_nums), 3)
        if "loc_error" in results.keys():
            for i, th in enumerate(self.conf.num_lines_th):
                cur_nums = list(map(lambda x: x[i], results["loc_error"]))
                summaries[f"loc_error@{th}lines"] = round(np.median(cur_nums), 3)

        figures = {}

        return summaries, figures, results


if __name__ == "__main__":
    dataset_name = Path(__file__).stem
    parser = get_eval_parser()
    args = parser.parse_intermixed_args()

    default_conf = OmegaConf.create(HPatchesPipeline.default_conf)

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

    pipeline = HPatchesPipeline(conf)


    #TODO: IMplement the grid search algorithm here

    #model = load_model(pipeline.conf.model, None)

    # py::arg("img"),
    # py::arg("scale") = 0.8,
    # py::arg("sigma_scale") = 0.6,
    # py::arg("density_th") = 0.0,
    # py::arg("quant") = 2.0,
    # py::arg("ang_th") = 22.5,
    # py::arg("with_gaussian") = false,
    # py::arg("gradnorm") = py::array(),
    # py::arg("gradangle") = py::array(),
    # py::arg("grad_nfa") = false);

    with_gaussian_params = [True]
    scale_params = [1.0]
    sigma_scale_params = [ 0.6, 0.575, 0.55]
    quant_params = [1.00, 1.10, 1.25, 1.5, 1,75]
    angle_th_params = [25.5, 26, 26.5]

    # Best configuration
    # {'name': 'lines.lsd', 'faster_lsd': True, 'search': True, 'with_gaussian': True, 'scale': 1.0, 'sigma_scale': 0.575, 'quant': 1.0, 'angle_th': 25.5}

    test = False
    if test:
        with_gaussian_params = [True, False]
        scale_params = [1.0, 0.8]
        sigma_scale_params = [0.6]
        quant_params = [2.0]
        angle_th_params = [22.5]


    # Generate all combinations
    all_combinations = list(itertools.product(
        with_gaussian_params,
        scale_params,
        sigma_scale_params,
        quant_params,
        angle_th_params
    ))

    total_combinaisons = len(all_combinations)

    def get_final_score(s):
        loc_error = s['loc_error@10lines'] + s['loc_error@50lines'] + s['loc_error@300lines']
        hM_error = s['mH_err@1'] + s['mH_err@3'] + s['mH_err@5']
        repeatability = s['repeatability@1px'] + s['repeatability@3px'] + s['repeatability@5px']
        # - ==> the higher the better
        return loc_error - hM_error - repeatability


    best = 100000
    best_type = {}
    best_s = {}

    pipeline.conf.model.extractor.search = True
    for counter, combo in enumerate(all_combinations):


        pipeline.conf.model.extractor.with_gaussian = combo[0]
        pipeline.conf.model.extractor.scale = combo[1]
        pipeline.conf.model.extractor.sigma_scale = combo[2]
        pipeline.conf.model.extractor.quant = combo[3]
        pipeline.conf.model.extractor.angle_th = combo[4]

        model = load_model(pipeline.conf.model, None)

        s, _, _ = pipeline.run(
            experiment_dir,
            model=model,
            overwrite=args.overwrite,
            overwrite_eval=args.overwrite_eval,
            plot=args.plot,
        )   

        if get_final_score(s) < best:
            best_type = copy.copy(pipeline.conf)
            best_s = copy.deepcopy(s)
            best = get_final_score(s)


        print(f"=== [{counter}/{total_combinaisons}] ===")
        print("=== Current configuration and score === ")
        pprint(combo)
        pprint(s)
        print("=== Best configuration===")
        pprint(best_type)
        pprint(best_s)

        print(f"Scores: [CURRENT] {get_final_score(s)} [BEST] {get_final_score(best_s)}")


    print("Running base configuration")
    pipeline.conf.model.extractor.search = False
    base_s, _,_ = pipeline.run(
            experiment_dir,
            overwrite=args.overwrite,
            overwrite_eval=args.overwrite_eval,
            plot=args.plot,
        )

    print("=== Base Configuration ===")
    pprint(base_s)

    print("=== Best Configuration ===")
    pprint(best_type)

    print(f"With improvement: {get_final_score(base_s) - best} and configuration: ")
    pprint(best_s)