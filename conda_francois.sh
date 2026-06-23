source /matx/u/franc141/miniforge3/etc/profile.d/conda.sh
conda activate glue




conda deactivate

conda create -n glue python=3.10 -y
conda activate glue

conda install -c conda-forge cmake ninja eigen suitesparse ceres-solver -y

pip install poselib
pip install omegaconf

python3 -m pip install -e .
./install.sh
./fetch_external.sh
pip install "git+https://github.com/rpautrat/homography_est.git@17b200d528e6aa8ac61a878a29265bf5f9d36c41"


conda install -c conda-forge \
    "cmake<4" \
    ninja \
    eigen \
    metis \
    "suitesparse=5.10.*" \
    "ceres-solver=2.2.*" \
    -y

export CMAKE_PREFIX_PATH=$CONDA_PREFIX
export Ceres_DIR=$CONDA_PREFIX/lib/cmake/Ceres
pip install --no-build-isolation \
"git+https://github.com/rpautrat/homography_est.git@17b200d528e6aa8ac61a878a29265bf5f9d36c41"


 python -m gluefactory.eval.hpatches_lines --conf gluefactory/configs/eval/linea+LM.yaml





python -m gluefactory.eval.inspect hpatches superpoint+NN_156644 \
  --default_plot matches

python -m gluefactory.eval.inspect hpatches superpoint+NN_156644 \
  --default_plot matches \
  --backend webagg


  