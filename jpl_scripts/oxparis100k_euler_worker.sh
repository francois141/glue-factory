#!/bin/bash
# Cmd params 'run_training_euler.sh [exp_name] [path to conf]'
 
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --gres=gpumem:23g
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=6000

CHUNK_VALUE=$1

if [ -z "$CHUNK_VALUE" ]; then
  echo "Error: Please provide a chunk value as the first argument."
  exit 1
fi

module load eth_proxy

module load stack/2024-06
module load python_cuda/3.11.6
module load cmake/3.27.7
module load eigen/3.4.0
module load ceres-solver/2.2.0
module load glog/0.6.0-sx7hlp6
module load gflags/2.2.2-gpd4lxs

source /cluster/home/fcosta/myenv/bin/activate

echo "Running chunk $CHUNK_VALUE"
python -m gluefactory.ground_truth_generation.oxparis_100k oxford_paris_mini_100k --num_H 100 --n_jobs 2 --n_jobs_dataloader 2 --chunk $CHUNK_VALUE
