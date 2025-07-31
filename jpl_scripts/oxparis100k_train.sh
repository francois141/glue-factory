#!/bin/bash
# Cmd params 'run_training_euler.sh [exp_name] [path to conf]'
 
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --gres=gpumem:23g
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=6000

DIR=$SLURM_SUBMIT_DIR/jpl_scripts
source $DIR/common.sh

SetupStack

echo "Starting training"
python -m gluefactory.train TRAIN_100k_VIT_ALL_PIXELS_5 --conf=gluefactory/configs/train_jpl_oxparis_100k_vit_5_threshold.yaml --distributed