#!/bin/bash
# Cmd params 'run_training_euler.sh [exp_name] [path to conf]'

#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --gres=gpumem:23g
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=6000

DIR=$SLURM_SUBMIT_DIR/jpl_scripts
source $DIR/common.sh

SetupStack

mkdir -p $DIR/eval/ablation
OUTPUT_DIR=$DIR/eval/ablation

echo "Evaluation of DeepLSD"
python -m gluefactory.eval.hpatches_lines --conf gluefactory/configs/eval/deeplsd+AF+LM.yaml --overwrite > "${OUTPUT_DIR}/ablation_deeplsd.txt"

echo "Evaluation of DeepLSD no AF"
python -m gluefactory.eval.hpatches_lines --conf gluefactory/configs/eval/deeplsd+LM.yaml --overwrite > "${OUTPUT_DIR}/ablation_deeplsd_no_af.txt"

echo "Evaluation of LSD"
python -m gluefactory.eval.hpatches_lines --conf gluefactory/configs/eval/lsd+LM.yaml --overwrite > "${OUTPUT_DIR}/ablation_lsd.txt"

echo "Evaluation of LSD + stride 2 + subset"
python -m gluefactory.eval.hpatches_lines --conf gluefactory/configs/eval/fastlsd+LM.yaml --overwrite > "${OUTPUT_DIR}/ablation_lsd_stride2_subset.txt"

echo "Evaluation of LSD + stride 2 + subset + GPU point selection"
python -m gluefactory.eval.hpatches_lines --conf gluefactory/configs/eval/lsd_points+LM.yaml --overwrite > "${OUTPUT_DIR}/ablation_lsd_from_points.txt"