from pathlib import Path
import os

def is_euler_cluster():
    hostname = os.environ.get('HOSTNAME', '')
    return ("eu" in hostname.lower())

root = Path(f"/cluster/scratch/{os.environ['USER']}/") if is_euler_cluster() else Path(__file__).parent.parent   # top-level directory

DATA_PATH = root / "data"  # datasets and pretrained weights
TRAINING_PATH = root / "outputs/training/"  # training checkpoints
EVAL_PATH = root / "outputs/results/"  # evaluation results
