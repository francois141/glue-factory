#!/bin/bash
# Cmd params 'run_training_euler.sh [exp_name] [path to conf]'

#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=6000

DIR=$SLURM_SUBMIT_DIR/jpl_scripts
source $DIR/common.sh

SetupStack

echo "Running all checkpoint JPL benchmark"

CONFIG_PATH="$1"

if [ -z "$CONFIG_PATH" ]; then
  echo "Usage: $0 <config_path.yaml>"
  exit 1
fi

CONFIG_NAME=$(basename "$CONFIG_PATH" .yaml)

# Set the target directory
TARGET_DIR=/cluster/scratch/$USER/outputs/training/$2


python -m gluefactory.eval.hpatches \
  --conf $CONFIG_PATH \
  --checkpoint /cluster/scratch/$USER/outputs/training/$2/checkpoint_best.tar \
  --overwrite 

# Print all .tar files in the directory (non-recursively)
for checkpoint in "$TARGET_DIR"/*.tar; do
    echo "Running current checkpoint: $checkpoint"

    python -m gluefactory.eval.hpatches \
    --conf $CONFIG_PATH \
    --checkpoint $checkpoint \
    --overwrite 
done
