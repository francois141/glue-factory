#!/bin/bash
# Cmd params 'run_training_euler.sh [exp_name] [path to conf]'

#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --gres=gpumem:23g
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=6000

DIR=$SLURM_SUBMIT_DIR/jpl_scripts
source $DIR/common.sh

SetupStack

mkdir -p $DIR/eval/performance
OUTPUT_DIR=$DIR/eval/performance

lscpu > "${OUTPUT_DIR}/cpu.txt"
nvidia-smi -q > "${OUTPUT_DIR}/gpu.txt"

NUMER_RUNS_GPU=500

echo "Profiling JPL"

python -m gluefactory.eval.timing_measurement --conf=gluefactory/configs/timing_eval/jpl.yaml --num_s=$NUMER_RUNS_GPU --device=cuda > "${OUTPUT_DIR}/jpl_gpu.txt"

echo "Profiling Wireframe (suarez)"

python -m gluefactory.eval.timing_measurement --conf=gluefactory/configs/timing_eval/wireframe.yaml --num_s=$NUMER_RUNS_GPU --device=cuda > "${OUTPUT_DIR}/suarez_gpu.txt"

echo "Profiling DeepLSD + Aliked"

python -m gluefactory.eval.timing_measurement --conf=gluefactory/configs/timing_eval/aliked+deeplsd.yaml --num_s=$NUMER_RUNS_GPU --device=cuda > "${OUTPUT_DIR}/aliked+deeplsd_gpu.txt"

echo "Profiling DeepLSD + Superpoint"

python -m gluefactory.eval.timing_measurement --conf=gluefactory/configs/timing_eval/superpoint+deeplsd.yaml --num_s=$NUMER_RUNS_GPU --device=cuda > "${OUTPUT_DIR}/superpoint+deeplsd_gpu.txt"

echo "Profiling DaD + ScaleLSD"

python -m gluefactory.eval.timing_measurement --conf=gluefactory/configs/timing_eval/dad+scalelsd.yaml --num_s=$NUMER_RUNS_GPU --device=cuda > "${OUTPUT_DIR}/dad+scalelsd_gpu.txt"

echo "Running measurements for ablations"

echo "Running DeepLSD with AF"

python -m gluefactory.eval.timing_measurement --conf gluefactory/configs/timing_eval/lines_only_deeplsd+AF.yaml --num_s=$NUMER_RUNS_GPU --device=cuda > "${OUTPUT_DIR}/ablation_deeplsd_af.txt"

echo "Running DeepLSD without AF"

python -m gluefactory.eval.timing_measurement --conf gluefactory/configs/timing_eval/lines_only_deeplsd.yaml --num_s=$NUMER_RUNS_GPU --device=cuda > "${OUTPUT_DIR}/ablation_deeplsd_without_af.txt"

echo "Running LSD"

python -m gluefactory.eval.timing_measurement --conf gluefactory/configs/timing_eval/lines_only_lsd.yaml --num_s=$NUMER_RUNS_GPU --device=cuda > "${OUTPUT_DIR}/ablation_lsd.txt"

echo "Running fast LSD"

python -m gluefactory.eval.timing_measurement --conf gluefactory/configs/timing_eval/lines_only_fastlsd.yaml --num_s=$NUMER_RUNS_GPU --device=cuda > "${OUTPUT_DIR}/ablation_fast_lsd.txt"

echo "Running gpu LSD"

python -m gluefactory.eval.timing_measurement --conf gluefactory/configs/timing_eval/lines_only_lsd_points.yaml --num_s=$NUMER_RUNS_GPU --device=cuda > "${OUTPUT_DIR}/ablation_gpu_lsd.txt"
