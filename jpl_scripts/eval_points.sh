#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6000

# CONFIG #
OVERWRITE_EVAL=true
OVERWRITE_EXTRACT=true
TIMING_ONLY=false
# END #

overwrite_flag=""
overwrite_eval_flag=""
timing_only_flag=""
if [ "$OVERWRITE_EVAL" == "true" ]; then
  echo "Overwrite eval results - recompute eval based on existing features!"
  overwrite_eval_flag="--overwrite_eval"
fi
if [ "$OVERWRITE_EXTRACT" == "true" ]; then
  echo "Overwrite features and eval."
  overwrite_flag="--overwrite"
fi
if [ "$TIMING_ONLY" == "true" ]; then
  echo "Timing only mode - skip evaluation, only measure inference time."
  timing_only_flag="--timing_only"
fi
DIR=$SLURM_SUBMIT_DIR/jpl_scripts
#source $DIR/common.sh

#SetupStack
#source $DIR/common.sh

#SetupStack
module load eth_proxy

source ~/miniconda3/etc/profile.d/conda.sh
conda activate jpl__env

echo "Running all checkpoint JPL benchmark"

mkdir -p $DIR/eval/point_evaluation
OUTPUT_DIR=$DIR/eval/point_evaluation

# -------------- Configurable Experiments -----------------
# Comment out experiments you don't want to run
EXPERIMENTS=(
  "megadepth"
  "hpatches"
  "scannet"
)
# ---------------------------------------------------------

# -------------- Function to Run Eval -----------------
run_eval() {
  local conf_path="$1"

  # Extract descriptor name
  local conf_file=$(basename "$conf_path" .yaml)
  local tag=${conf_file//+/_}  # Replace + with _ for filename

  echo "Running evaluation for: $tag"

  for experiment in "${EXPERIMENTS[@]}"; do
    case "$experiment" in
      "megadepth")
        python -m gluefactory.eval.megadepth1500 \
          --conf "$conf_path" \
          $overwrite_eval_flag $overwrite_flag $timing_only_flag > "${OUTPUT_DIR}/${tag}_megadepth.txt"
        ;;
      "scannet")
        python -m gluefactory.eval.scannet1500 \
          --conf "$conf_path" \
          $overwrite_eval_flag $overwrite_flag $timing_only_flag > "${OUTPUT_DIR}/${tag}_scannet1500.txt"
        ;;
      "hpatches")
        python -m gluefactory.eval.hpatches \
          --conf "$conf_path" \
          $overwrite_eval_flag $overwrite_flag $timing_only_flag > "${OUTPUT_DIR}/${tag}_hpatches.txt"
        ;;
      *)
        echo "Warning: Unknown experiment '$experiment'"
        ;;
    esac
  done
}
# -----------------------------------------------------


# Run evaluations
# NN evaluation
run_eval "./gluefactory/configs/eval/plnet+NN_points_evaluation.yaml"
run_eval "./gluefactory/configs/eval/jpl+NN_points_evaluation.yaml"
run_eval "./gluefactory/configs/eval/superpoint+NN.yaml"
run_eval "./gluefactory/configs/eval/aliked+NN.yaml"
run_eval "./gluefactory/configs/eval/disk+NN.yaml"
run_eval "./gluefactory/configs/eval/sift+NN.yaml"
run_eval "./gluefactory/configs/eval/dedode+NN.yaml"
run_eval "./gluefactory/configs/eval/dad+NN.yaml"
run_eval "./gluefactory/configs/eval/xfeat+NN.yaml"
run_eval "./gluefactory/configs/eval/suarez+NN_points_evaluation.yaml"

# ROMA evaluation
#run_eval "./gluefactory/configs/eval/jpl+ROMA_points_evaluation.yaml"
#run_eval "./gluefactory/configs/eval/superpoint+ROMA.yaml"
#run_eval "./gluefactory/configs/eval/aliked+ROMA.yaml"
#run_eval "./gluefactory/configs/eval/disk+ROMA.yaml"
#run_eval "./gluefactory/configs/eval/sift+ROMA.yaml"
#run_eval "./gluefactory/configs/eval/dedode+ROMA.yaml"
#run_eval "./gluefactory/configs/eval/dad+ROMA.yaml"
#run_eval "./gluefactory/configs/eval/xfeat+ROMA.yaml"
