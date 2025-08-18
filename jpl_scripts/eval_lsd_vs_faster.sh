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

mkdir -p $DIR/eval/fast_lsd_eval

OUTPUT_DIR=$DIR/eval/fast_lsd_eval


echo "Evaluation on HPatches Dataset - raw lsd vs fast lsd"

python -m gluefactory.eval.hpatches_lines --conf gluefactory/configs/eval/lsd+LM.yaml --overwrite > "${OUTPUT_DIR}/hpatches_raw_lsd.txt"
python -m gluefactory.eval.hpatches_lines --conf gluefactory/configs/eval/fastlsd+LM.yaml --overwrite > "${OUTPUT_DIR}/hpatches_raw_lsd_fast.txt"
python -m gluefactory.eval.hpatches_lines --conf gluefactory/configs/eval/lsd_opt+LM.yaml --overwrite > "${OUTPUT_DIR}/hpatches_raw_lsd_optimal.txt"
python -m gluefactory.eval.hpatches_lines --conf gluefactory/configs/eval/lsd_points+LM.yaml --overwrite > "${OUTPUT_DIR}/hpatches_raw_lsd_points.txt"

echo "Evaluation on RDNIM Dataset - raw lsd vs fast lsd"

python -m gluefactory.eval.rdnim_lines --conf gluefactory/configs/eval/lsd+LM.yaml --overwrite > "${OUTPUT_DIR}/rdnim_lines_raw_lsd.txt"
python -m gluefactory.eval.rdnim_lines --conf gluefactory/configs/eval/fastlsd+LM.yaml --overwrite > "${OUTPUT_DIR}/rdnim_lines_raw_lsd_fast.txt"
python -m gluefactory.eval.rdnim_lines --conf gluefactory/configs/eval/lsd_opt+LM.yaml --overwrite > "${OUTPUT_DIR}/rdnim_lines_raw_lsd_optimal.txt"
python -m gluefactory.eval.rdnim_lines --conf gluefactory/configs/eval/lsd_points+LM.yaml --overwrite > "${OUTPUT_DIR}/rdnim_lines_raw_lsd_points.txt"