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

mkdir $DIR/eval/
mkdir $DIR/eval/af_evaluation

OUTPUT_DIR=$DIR/eval/af_evaluation

echo "Evaluation of deeplsd without angle field"

python -m gluefactory.eval.hpatches_lines --conf gluefactory/configs/eval/deeplsd+LM.yaml --overwrite > "${OUTPUT_DIR}/hpatches_deeplsd_without_af.txt"
python -m gluefactory.eval.rdnim_lines --conf gluefactory/configs/eval/deeplsd+LM.yaml --overwrite > "${OUTPUT_DIR}/rdnim_lines_deeplsd_without_af.txt"

echo "Evaluation of deeplsd with angle field"

python -m gluefactory.eval.hpatches_lines --conf gluefactory/configs/eval/deeplsd+AF+LM.yaml --overwrite > "${OUTPUT_DIR}/hpatches_deeplsd_with_af.txt"
python -m gluefactory.eval.rdnim_lines --conf gluefactory/configs/eval/deeplsd+AF+LM.yaml --overwrite > "${OUTPUT_DIR}/rdnim_lines_deeplsd_with_af.txt"

echo "Evaluation of deeplsd without angle field - with fast_lsd"

python -m gluefactory.eval.hpatches_lines --conf gluefactory/configs/eval/deeplsd+fastlsd+LM.yaml --overwrite > "${OUTPUT_DIR}/hpatches_deeplsd+fastlsd_without_af.txt"
python -m gluefactory.eval.rdnim_lines --conf gluefactory/configs/eval/deeplsd+fastlsd+LM.yaml --overwrite > "${OUTPUT_DIR}/rdnim_lines_deeplsd+fastlsd_without_af.txt"

echo "Evaluation of deeplsd with angle field - with fast_lsd"

python -m gluefactory.eval.hpatches_lines --conf gluefactory/configs/eval/deeplsd+fastlsd+AF+LM.yaml --overwrite > "${OUTPUT_DIR}/hpatches_deeplsd+fastlsd_with_af.txt"
python -m gluefactory.eval.rdnim_lines --conf gluefactory/configs/eval/deeplsd+fastlsd+AF+LM.yaml --overwrite > "${OUTPUT_DIR}/rdnim_lines_deeplsd+fastlsd_with_af.txt"