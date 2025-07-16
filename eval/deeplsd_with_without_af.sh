#!/bin/bash
# Cmd params 'run_training_euler.sh [exp_name] [path to conf]'

#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --gres=gpumem:23g
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=6000

if [[ "$(hostname)" == eu* ]]; then
  module load eth_proxy

  module load stack/2024-06
  module load python_cuda/3.11.6
  module load cmake/3.27.7
  module load eigen/3.4.0
  module load ceres-solver/2.2.0
  module load glog/0.6.0-sx7hlp6
  module load gflags/2.2.2-gpd4lxs

  source /cluster/home/fcosta/myenv/bin/activate
fi

mkdir eval/af_evaluation

echo "Evaluation of deeplsd without angle field"

python -m gluefactory.eval.hpatches_lines --conf gluefactory/configs/eval/deeplsd+LM.yaml --overwrite > "eval/af_evaluation/hpatches_deeplsd_without_af.txt"
python -m gluefactory.eval.rdnim_lines --conf gluefactory/configs/eval/deeplsd+LM.yaml --overwrite > "eval/af_evaluation/rdnim_lines_deeplsd_without_af.txt"

echo "Evaluation of deeplsd with angle field"

python -m gluefactory.eval.hpatches_lines --conf gluefactory/configs/eval/deeplsd+AF+LM.yaml --overwrite > "eval/af_evaluation/hpatches_deeplsd_with_af.txt"
python -m gluefactory.eval.rdnim_lines --conf gluefactory/configs/eval/deeplsd+AF+LM.yaml --overwrite > "eval/af_evaluation/rdnim_lines_deeplsd_with_af.txt"

echo "Evaluation of deeplsd without angle field - with fast_lsd"

python -m gluefactory.eval.hpatches_lines --conf gluefactory/configs/eval/deeplsd+fastlsd+LM.yaml --overwrite > "eval/af_evaluation/hpatches_deeplsd+fastlsd_without_af.txt"
python -m gluefactory.eval.rdnim_lines --conf gluefactory/configs/eval/deeplsd+fastlsd+LM.yaml --overwrite > "eval/af_evaluation/rdnim_lines_deeplsd+fastlsd_without_af.txt"

echo "Evaluation of deeplsd with angle field - with fast_lsd"

python -m gluefactory.eval.hpatches_lines --conf gluefactory/configs/eval/deeplsd+fastlsd+AF+LM.yaml --overwrite > "eval/af_evaluation/hpatches_deeplsd+fastlsd_with_af.txt"
python -m gluefactory.eval.rdnim_lines --conf gluefactory/configs/eval/deeplsd+fastlsd+AF+LM.yaml --overwrite > "eval/af_evaluation/rdnim_lines_deeplsd+fastlsd_with_af.txt"