#!/bin/bash
# RUN BENCHMARK ON EULER
# Cmd params 'run_eval_hpatches.sh [path to conf/conf name]'
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --gres=gpumem:23g
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8000
#SBATCH --output=eval.out
#SBATCH --mail-type=END
#SBATCH --mail-user=r.kreft@stud.ethz.ch
#SBATCH --job-name="jpl_benchmark"

module load eth_proxy

echo "Conf-Path: $1"

# Print SLURM variables so you see how your resources are allocated
echo "[sbatch-master] Job Name: $SLURM_JOB_NAME"
echo "[sbatch-master] Job ID: $SLURM_JOB_ID"
echo "[sbatch-master] Num Nodes: $SLURM_NNODES"
echo "[sbatch-master] Allocated Node(s): $SLURM_NODELIST"
echo "[sbatch-master] Number of Tasks: $SLURM_NTASKS"
echo "[sbatch-master] MasterNodeID: $SLURM_NODEID"
echo "[sbatch-master] Number of GPUs allocated: $SLURM_GPUS_ON_NODE"
echo "Current path: $(pwd)"
echo "Current user: $(whoami)"

#source ~/jpl_venv/bin/activate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate jpl__env
cd ~/glue-factory || exit

# !! if copying this script as a template, change experiment name and path to config(create new config) !!
# Run script (adapt distributed and restore if needed)
python -m gluefactory.eval.hpatches_lines --conf="$1" --overwrite
echo "Finished Benchmark!"
