readonly VENV_PATH=/cluster/home/fcosta/myenv

function SetupStack() {
  if [[ "$(hostname)" == eu* ]]; then
    module load eth_proxy

    module load stack/2024-06
    module load python_cuda/3.11.6
    module load cmake/3.27.7
    module load eigen/3.4.0
    module load ceres-solver/2.2.0
    module load glog/0.6.0-sx7hlp6
    module load gflags/2.2.2-gpd4lxs
  fi

  source $VENV_PATH/bin/activate

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
}

