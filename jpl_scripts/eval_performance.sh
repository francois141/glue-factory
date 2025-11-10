#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6000

DIR=$SLURM_SUBMIT_DIR/jpl_scripts
source $DIR/common.sh

SetupStack

mkdir -p $DIR/eval/performance
OUTPUT_DIR=$DIR/eval/performance

lscpu > "${OUTPUT_DIR}/cpu.txt"
nvidia-smi -q > "${OUTPUT_DIR}/gpu.txt"

NUMER_RUNS_GPU=500

# List of configs to profile - comment out any you don't want to run
# Format: "Display Name|full_config_path|output_filename"
CONFIGS=(
    "Profiling JPL|gluefactory/configs/timing_eval/jpl.yaml|jpl_gpu.txt"
    "Profiling Wireframe (suarez)|gluefactory/configs/timing_eval/wireframe.yaml|suarez_gpu.txt"
    "Profiling DeepLSD + Aliked|gluefactory/configs/timing_eval/aliked+deeplsd.yaml|aliked+deeplsd_gpu.txt"
    "Profiling DeepLSD + Superpoint|gluefactory/configs/timing_eval/superpoint+deeplsd.yaml|superpoint+deeplsd_gpu.txt"
    "Profiling DaD + ScaleLSD|gluefactory/configs/timing_eval/dad+scalelsd.yaml|dad+scalelsd_gpu.txt"
    "Running DeepLSD with AF|gluefactory/configs/timing_eval/lines_only_deeplsd+AF.yaml|ablation_deeplsd_af.txt"
    "Running DeepLSD without AF|gluefactory/configs/timing_eval/lines_only_deeplsd.yaml|ablation_deeplsd_without_af.txt"
    "Running LSD|gluefactory/configs/timing_eval/lines_only_lsd.yaml|ablation_lsd.txt"
    "Running fast LSD|gluefactory/configs/timing_eval/lines_only_fastlsd.yaml|ablation_fast_lsd.txt"
    "Running gpu LSD|gluefactory/configs/timing_eval/lines_only_lsd_points.yaml|ablation_gpu_lsd.txt"
)

# Run profiling
for config_entry in "${CONFIGS[@]}"; do
    IFS='|' read -r display_name config_path output_file <<< "$config_entry"
    echo "$display_name"
    python -m gluefactory.eval.timing_measurement --conf="$config_path" --num_s=$NUMER_RUNS_GPU --device=cuda > "${OUTPUT_DIR}/${output_file}"
done
