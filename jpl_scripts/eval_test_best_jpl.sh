#!/bin/bash
# Cmd params 'run_training_euler.sh [exp_name] [path to conf]'

#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=6000

DIR=$SLURM_SUBMIT_DIR/jpl_scripts
source $DIR/common.sh

SetupStack

echo "Running all checkpoint JPL benchmark"


CONFIG_NAME=$(basename "$CONFIG_PATH" .yaml)

# Set the target directory
TARGET_DIR=/cluster/scratch/$USER/outputs/training/$2


python -m gluefactory.eval.hpatches_lines \
  --conf /cluster/home/fcosta/glue-factory/gluefactory/configs/eval_aliked_backbone_0.575.yaml \
  --checkpoint /cluster/home/fcosta/glue-factory/assets/jpl_best.tar \
  --overwrite 

  python -m gluefactory.eval.hpatches_lines \
  --conf /cluster/home/fcosta/glue-factory/gluefactory/configs/eval_aliked_backbone_0.6.yaml \
  --checkpoint /cluster/home/fcosta/glue-factory/assets/jpl_best.tar \
  --overwrite 

  python -m gluefactory.eval.hpatches_lines \
  --conf /cluster/home/fcosta/glue-factory/gluefactory/configs/eval_aliked_backbone_0.65.yaml \
  --checkpoint /cluster/home/fcosta/glue-factory/assets/jpl_best.tar \
  --overwrite 

  exit(0)


python -m gluefactory.eval.hpatches_lines \
  --conf /cluster/home/fcosta/glue-factory/gluefactory/configs/eval_aliked_backbone_0.7.yaml \
  --checkpoint /cluster/home/fcosta/glue-factory/assets/jpl_best.tar \
  --overwrite 

python -m gluefactory.eval.hpatches_lines \
  --conf /cluster/home/fcosta/glue-factory/gluefactory/configs/eval_aliked_backbone_0.8.yaml \
  --checkpoint /cluster/home/fcosta/glue-factory/assets/jpl_best.tar \
  --overwrite 

python -m gluefactory.eval.hpatches_lines \
  --conf /cluster/home/fcosta/glue-factory/gluefactory/configs/eval_aliked_backbone_0.9.yaml \
  --checkpoint /cluster/home/fcosta/glue-factory/assets/jpl_best.tar \
  --overwrite 

python -m gluefactory.eval.hpatches_lines \
  --conf /cluster/home/fcosta/glue-factory/gluefactory/configs/eval_aliked_backbone_1.0.yaml \
  --checkpoint /cluster/home/fcosta/glue-factory/assets/jpl_best.tar \
  --overwrite 

python -m gluefactory.eval.hpatches_lines \
  --conf /cluster/home/fcosta/glue-factory/gluefactory/configs/eval_aliked_backbone_1.1.yaml \
  --checkpoint /cluster/home/fcosta/glue-factory/assets/jpl_best.tar \
  --overwrite 

python -m gluefactory.eval.hpatches_lines \
  --conf /cluster/home/fcosta/glue-factory/gluefactory/configs/eval_aliked_backbone_1.2.yaml \
  --checkpoint /cluster/home/fcosta/glue-factory/assets/jpl_best.tar \
  --overwrite 

