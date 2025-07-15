mkdir eval/fast_lsd_eval

echo "Evaluation on HPatches Dataset - raw lsd vs fast lsd"

python -m gluefactory.eval.hpatches_lines --conf gluefactory/configs/eval/lsd+LM.yaml --overwrite > "eval/fast_lsd_eval/hpatches_raw_lsd.txt"
python -m gluefactory.eval.hpatches_lines --conf gluefactory/configs/eval/fastlsd+LM.yaml --overwrite > "eval/fast_lsd_eval/hpatches_raw_fastlsd.txt"

echo "Evaluation on RDNIM Dataset - raw lsd vs fast lsd"

python -m gluefactory.eval.rdnim_lines --conf gluefactory/configs/eval/lsd+LM.yaml --overwrite > "eval/fast_lsd_eval/rdnim_lines_raw_lsd.txt"
python -m gluefactory.eval.rdnim_lines --conf gluefactory/configs/eval/fastlsd+LM.yaml --overwrite > "eval/fast_lsd_eval/rdnim_lines_raw_fastlsd.txt"

echo "Evaluation on Hpatches Dataset - deep lsd with lsd vs fast lsd"

python -m gluefactory.eval.hpatches_lines --conf gluefactory/configs/eval/deeplsd+LM.yaml --overwrite > "eval/fast_lsd_eval/hpatches_deeplsd.txt"
python -m gluefactory.eval.hpatches_lines --conf gluefactory/configs/eval/deeplsd+fastlsd+LM.yaml --overwrite > "eval/fast_lsd_eval/hpatches_deeplsd+fastlsd.txt"

echo "Evaluation on RDNIM Dataset - deep lsd with lsd vs fast lsd"

python -m gluefactory.eval.rdnim_lines --conf gluefactory/configs/eval/deeplsd+LM.yaml --overwrite > "eval/fast_lsd_eval/rdnim_lines_deeplsd.txt"
python -m gluefactory.eval.rdnim_lines --conf gluefactory/configs/eval/deeplsd+fastlsd+LM.yaml --overwrite > "eval/fast_lsd_eval/rdnim_lines_deeplsd+fastlsd.txt"