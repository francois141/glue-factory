#!/bin/bash
# Cmd params 'run_training_euler.sh [exp_name] [path to conf]'
 
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --gres=gpumem:23g
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=6000

DIR=$SLURM_SUBMIT_DIR/jpl_scripts
source $DIR/common.sh

SetupStack

CONFIG_PATH="$1"

if [ -z "$CONFIG_PATH" ]; then
  echo "Usage: $0 <config_path.yaml>"
  exit 1
fi

CONFIG_NAME=$(basename "$CONFIG_PATH" .yaml)

echo "Starting training with config: $CONFIG_NAME"
python -m gluefactory.train "$CONFIG_NAME" --conf="$CONFIG_PATH" --distributed