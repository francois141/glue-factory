#!/bin/bash
# Cmd params 'run_training_euler.sh [exp_name] [path to conf]'

#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=6000

DIR=$SLURM_SUBMIT_DIR/jpl_scripts
source $DIR/common.sh

SetupStack

echo "Running all checkpoint JPL benchmark"

mkdir -p $DIR/eval/point_evaluation
OUTPUT_DIR=$DIR/eval/point_evaluation

# -------------- Function to Run Eval -----------------
run_eval() {
  echo "Bonjour"
  local conf_path="$1"
  local checkpoint="$2"

  # Extract descriptor name
  local conf_file=$(basename "$conf_path" .yaml)
  local tag=${conf_file//+/_}  # Replace + with _ for filename

  echo "Running evaluation for: $tag"

  python -m gluefactory.eval.megadepth1500 \
    --conf "$conf_path" \
    --overwrite > "${OUTPUT_DIR}/${tag}_megadepth.txt"

  python -m gluefactory.eval.scannet1500 \
    --conf "$conf_path" \
    --overwrite > "${OUTPUT_DIR}/${tag}_scannet1500.txt"

  python -m gluefactory.eval.hpatches \
    --conf "$conf_path" \
    --overwrite > "${OUTPUT_DIR}/${tag}_hpatches.txt"
}
# -----------------------------------------------------

# Run evaluations
# NN evaluation
run_eval "./gluefactory/configs/eval/superpoint+NN.yaml" "$CHECKPOINT"
run_eval "./gluefactory/configs/eval/aliked+NN.yaml" "$CHECKPOINT"
run_eval "./gluefactory/configs/eval/disk+NN.yaml" "$CHECKPOINT"
run_eval "./gluefactory/configs/eval/sift+NN.yaml" "$CHECKPOINT"
# ROMA evaluation
run_eval "./gluefactory/configs/eval/superpoint+ROMA.yaml" "$CHECKPOINT"
run_eval "./gluefactory/configs/eval/aliked+ROMA.yaml" "$CHECKPOINT"
run_eval "./gluefactory/configs/eval/disk+ROMA.yaml" "$CHECKPOINT"
run_eval "./gluefactory/configs/eval/sift+ROMA.yaml" "$CHECKPOINT"
run_eval "./gluefactory/configs/eval/dad+ROMA.yaml" "$CHECKPOINT"