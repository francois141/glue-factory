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

mkdir -p $DIR/eval/params_lsd

OUTPUT_FILE=$DIR/eval/params_lsd/output.txt

echo "Running search space on faster_lsd"

python -m gluefactory.eval.hpatches_lines_grid_search --conf gluefactory/configs/eval/fastlsd+LM.yaml --overwrite > $OUTPUT_FILE