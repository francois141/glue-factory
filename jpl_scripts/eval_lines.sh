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

mkdir -p $DIR/eval/line_evaluation
OUTPUT_DIR=$DIR/eval/line_evaluation

echo "Evaluation of JPL"

python -m gluefactory.eval.hpatches_lines --conf gluefactory/configs/eval/jpl+LM.yaml --overwrite > "${OUTPUT_DIR}/hpatches_jpl.txt"
python -m gluefactory.eval.rdnim_lines --conf gluefactory/configs/eval/jpl+LM.yaml --overwrite > "${OUTPUT_DIR}/rdnim_lines_jpl.txt"

echo "Evaluation of JPL with ferrari LSD"

python -m gluefactory.eval.hpatches_lines --conf gluefactory/configs/eval/jpl+points_lsd+LM.yaml --overwrite > "${OUTPUT_DIR}/hpatches_jpl_ferrari_lsd.txt"
python -m gluefactory.eval.rdnim_lines --conf gluefactory/configs/eval/jpl+points_lsd+LM.yaml --overwrite > "${OUTPUT_DIR}/rdnim_lines_jpl_ferrari_lsd.txt"

echo "Evaluation of deeplsd"

python -m gluefactory.eval.hpatches_lines --conf gluefactory/configs/eval/deeplsd+AF+LM.yaml --overwrite > "${OUTPUT_DIR}/hpatches_deeplsd.txt"
python -m gluefactory.eval.rdnim_lines --conf gluefactory/configs/eval/deeplsd+AF+LM.yaml --overwrite > "${OUTPUT_DIR}/rdnim_lines_deeplsd.txt"

echo "Evaluation of deeplsd without angle field"

python -m gluefactory.eval.hpatches_lines --conf gluefactory/configs/eval/deeplsd+LM.yaml --overwrite > "${OUTPUT_DIR}/hpatches_deeplsd_without_af.txt"
python -m gluefactory.eval.rdnim_lines --conf gluefactory/configs/eval/deeplsd+LM.yaml --overwrite > "${OUTPUT_DIR}/rdnim_lines_deeplsd_without_af.txt"

echo "Evaluation of scalelsd"

python -m gluefactory.eval.hpatches_lines --conf gluefactory/configs/eval/scalelsd+LM.yaml --overwrite > "${OUTPUT_DIR}/hpatches_scalelsd.txt"
python -m gluefactory.eval.rdnim_lines --conf gluefactory/configs/eval/scalelsd+LM.yaml --overwrite > "${OUTPUT_DIR}/rdnim_lines_scalelsd.txt"

echo "Evaluation of sold2"

python -m gluefactory.eval.hpatches_lines --conf gluefactory/configs/eval/sold2+LM.yaml --overwrite > "${OUTPUT_DIR}/hpatches_sold2.txt"
python -m gluefactory.eval.rdnim_lines --conf gluefactory/configs/eval/sold2+LM.yaml --overwrite > "${OUTPUT_DIR}/rdnim_lines_sold2.txt"

echo "Evaluation of lsd"

python -m gluefactory.eval.hpatches_lines --conf gluefactory/configs/eval/lsd+LM.yaml --overwrite > "${OUTPUT_DIR}/hpatches_lsd.txt"
python -m gluefactory.eval.rdnim_lines --conf gluefactory/configs/eval/lsd+LM.yaml --overwrite > "${OUTPUT_DIR}/rdnim_lines_lsd.txt"