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

mkdir -p $DIR/eval/jpl_benchmark

OUTPUT_DIR=$DIR/eval/jpl_benchmark

echo "Running JPL benchmark: $1"

ENTRY=$1
CHECKPOINT_PATH="/cluster/scratch/fcosta/outputs/training/$ENTRY/checkpoint_best.tar"

python -m gluefactory.eval.hpatches_lines \
  --conf gluefactory/configs/benchmark_jpl_lsd.yaml \
  --checkpoint $CHECKPOINT_PATH \
  --overwrite > "${OUTPUT_DIR}/${ENTRY}_jpl_lsd_benchmark.txt"

python -m gluefactory.eval.hpatches_lines \
  --conf gluefactory/configs/benchmark_jpl_fastlsd.yaml \
  --checkpoint $CHECKPOINT_PATH \
  --overwrite > "${OUTPUT_DIR}/${ENTRY}_jpl_fast_benchmark.txt"
