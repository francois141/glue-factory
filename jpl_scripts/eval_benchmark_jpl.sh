#!/bin/bash
# Cmd params 'run_training_euler.sh [exp_name] [path to conf]'

#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --gres=gpumem:23g
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=6000

DIR=$SLURM_SUBMIT_DIR/jpl_scripts
source $DIR/common.sh

SetupStack

mkdir $DIR/eval/
mkdir $DIR/eval/jpl_benchmark

OUTPUT_DIR=$DIR/eval/jpl_benchmark

echo "Running JPL benchmark"

python -m gluefactory.eval.hpatches_lines --conf gluefactory/configs/COMMON_BENCHMARK_CONF.yaml --overwrite > "${OUTPUT_DIR}/jpl_benchmark.txt"