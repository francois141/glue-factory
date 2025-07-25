#!/bin/bash
# Cmd params 'run_training_euler.sh [exp_name] [path to conf]'
 
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --gres=gpumem:23g
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=6000

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && pwd)"
source $DIR/common.sh

CHUNK_VALUE=$1

if [ -z "$CHUNK_VALUE" ]; then
  echo "Error: Please provide a chunk value as the first argument."
  exit 1
fi

SetupStack

echo "Running chunk $CHUNK_VALUE"
python -m gluefactory.ground_truth_generation.oxparis_100k oxford_paris_mini_100k --num_H 100 --n_jobs 2 --n_jobs_dataloader 2 --chunk $CHUNK_VALUE
