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

module load eth_proxy

source ~/miniconda3/etc/profile.d/conda.sh
conda activate jpl__env

mkdir -p $DIR/eval/line_evaluation
OUTPUT_DIR=$DIR/eval/line_evaluation

# List of benchmarks to run - comment out any you don't want to run
BENCHMARKS=(
    "hpatches_lines"
    "rdnim_lines"
)

# List of configs to evaluate - comment out any you don't want to run
# Format: "Display Name|full_config_path|output_suffix"
CONFIGS=(
    "PLNet|gluefactory/configs/eval/plnet+LM.yaml|plnet"
    "Suarez|gluefactory/configs/eval/suarez+LM.yaml|wireframe_suarez"
    "JPL|gluefactory/configs/eval/jpl+LM.yaml|jpl"
    "JPL with ferrari LSD|gluefactory/configs/eval/jpl+points_lsd+LM.yaml|jpl_ferrari_lsd"
    "deeplsd|gluefactory/configs/eval/deeplsd+AF+LM.yaml|deeplsd"
    "deeplsd without angle field|gluefactory/configs/eval/deeplsd+LM.yaml|deeplsd_without_af"
    "scalelsd|gluefactory/configs/eval/scalelsd+LM.yaml|scalelsd"
    "sold2|gluefactory/configs/eval/sold2+LM.yaml|sold2"
    "lsd|gluefactory/configs/eval/lsd+LM.yaml|lsd"
    "M-LSD|gluefactory/configs/eval/mlsd+LM.yaml|mlsd"
    "TP-LSD|gluefactory/configs/eval/tplsd+LM.yaml|tplsd"
    "ELSED|gluefactory/configs/eval/ELSED+LM.yaml|elsed"
)

# Run evaluations
for config_entry in "${CONFIGS[@]}"; do
    IFS='|' read -r display_name config_path output_suffix <<< "$config_entry"
    echo "Evaluation of $display_name"
    for benchmark in "${BENCHMARKS[@]}"; do
        python -m "gluefactory.eval.${benchmark}" --conf "$config_path" $overwrite_eval_flag $overwrite_flag $timing_only_flag > "${OUTPUT_DIR}/${benchmark}_${output_suffix}.txt"
    done
done
