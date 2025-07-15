mkdir eval/af_evaluation

echo "Evaluation of deeplsd without angle field"

python -m gluefactory.eval.hpatches_lines --conf gluefactory/configs/eval/deeplsd+LM.yaml --overwrite > "eval/af_evaluation/hpatches_deeplsd_without_af.txt"
python -m gluefactory.eval.rdnim_lines --conf gluefactory/configs/eval/deeplsd+LM.yaml --overwrite > "eval/af_evaluation/rdnim_lines_deeplsd_without_af.txt"

echo "Evaluation of deeplsd with angle field"

python -m gluefactory.eval.hpatches_lines --conf gluefactory/configs/eval/deeplsd+AF+LM.yaml --overwrite > "eval/af_evaluation/hpatches_deeplsd_with_af.txt"
python -m gluefactory.eval.rdnim_lines --conf gluefactory/configs/eval/deeplsd+AF+LM.yaml --overwrite > "eval/af_evaluation/rdnim_lines_deeplsd_with_af.txt"