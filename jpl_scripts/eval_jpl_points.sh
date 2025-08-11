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

mkdir -p $DIR/eval/jpl_points

OUTPUT_DIR=$DIR/eval/jpl_points

# Set the target directory
TARGET_DIR=/cluster/scratch/$USER/outputs/training/$2


# JPL
python -m gluefactory.eval.megadepth1500 \
  --conf $CONFIG_PATH \
  --checkpoint /cluster/scratch/$USER/outputs/training/$2/checkpoint_best.tar \
  --overwrite > "${OUTPUT_DIR}/jpl_megadepth.txt"

python -m gluefactory.eval.scannet1500 \
  --conf $CONFIG_PATH \
  --checkpoint /cluster/scratch/$USER/outputs/training/$2/checkpoint_best.tar \
  --overwrite > "${OUTPUT_DIR}/jpl_scannet1500.txt"

python -m gluefactory.eval.hpatches \
  --conf $CONFIG_PATH \
  --checkpoint /cluster/scratch/$USER/outputs/training/$2/checkpoint_best.tar \
  --overwrite > "${OUTPUT_DIR}/jpl_hpatches.txt"

# Superpoint
python -m gluefactory.eval.megadepth1500 \
  --conf ./gluefactory/configs/eval/superpoint+NN.yaml \
  --checkpoint /cluster/scratch/$USER/outputs/training/$2/checkpoint_best.tar \
  --overwrite > "${OUTPUT_DIR}/superpoint_megadepth.txt"

python -m gluefactory.eval.scannet1500 \
  --conf ./gluefactory/configs/eval/superpoint+NN.yaml \
  --checkpoint /cluster/scratch/$USER/outputs/training/$2/checkpoint_best.tar \
  --overwrite > "${OUTPUT_DIR}/superpoint_scannet1500.txt"

python -m gluefactory.eval.hpatches \
  --conf ./gluefactory/configs/eval/superpoint+NN.yaml \
  --checkpoint /cluster/scratch/$USER/outputs/training/$2/checkpoint_best.tar \
  --overwrite > "${OUTPUT_DIR}/superpoint_hpatches.txt"


# Aliked
python -m gluefactory.eval.megadepth1500 \
  --conf ./gluefactory/configs/eval/aliked+NN.yaml \
  --checkpoint /cluster/scratch/$USER/outputs/training/$2/checkpoint_best.tar \
  --overwrite > "${OUTPUT_DIR}/aliked_megadepth.txt"

python -m gluefactory.eval.scannet1500 \
  --conf ./gluefactory/configs/eval/aliked+NN.yaml \
  --checkpoint /cluster/scratch/$USER/outputs/training/$2/checkpoint_best.tar \
  --overwrite > "${OUTPUT_DIR}/aliked_scannet1500.txt"

python -m gluefactory.eval.hpatches \
  --conf ./gluefactory/configs/eval/aliked+NN.yaml \
  --checkpoint /cluster/scratch/$USER/outputs/training/$2/checkpoint_best.tar \
  --overwrite > "${OUTPUT_DIR}/aliked_hpatches.txt"