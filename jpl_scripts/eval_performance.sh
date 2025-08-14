#!/bin/bash
# Cmd params 'run_training_euler.sh [exp_name] [path to conf]'

#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --gres=gpumem:23g
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=6000

DIR=$SLURM_SUBMIT_DIR/jpl_scripts
source $DIR/common.sh

SetupStack

mkdir -p $DIR/eval/performance

OUTPUT_FILE=$DIR/eval/performance/output.txt

echo "Evaluate performance of the JPL model on Oxford Paris Mini dataset"

python -m gluefactory.eval.timing_measurement --conf=gluefactory/configs/timing_conf_aliked.yaml --num_s=100 --device=cuda > $OUTPUT_FILE
