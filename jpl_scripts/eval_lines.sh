#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --gres=gpumem:23g
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=6000

DIR=$SLURM_SUBMIT_DIR/jpl_scripts
source $DIR/common.sh

SetupStack

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
)

# Run evaluations
for config_entry in "${CONFIGS[@]}"; do
    IFS='|' read -r display_name config_path output_suffix <<< "$config_entry"
    echo "Evaluation of $display_name"
    for benchmark in "${BENCHMARKS[@]}"; do
        python -m "gluefactory.eval.${benchmark}" --conf "$config_path" --overwrite > "${OUTPUT_DIR}/${benchmark}_${output_suffix}.txt"
    done
done