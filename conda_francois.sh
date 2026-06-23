source /matx/u/franc141/miniforge3/etc/profile.d/conda.sh


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


 Hpatches

 {'extraction_runtime_avg_s': 0.05018,
 'extraction_runtime_total_s': 27.099,
 'loc_error@10lines': 0.947,
 'loc_error@300lines': 1.27,
 'loc_error@50lines': 1.27,
 'mH_err@1': 0.022,
 'mH_err@3': 0.156,
 'mH_err@5': 0.207,
 'mloc_error': 1.158,
 'mnum_lines': 7.0,
 'mrepeatability': 0.157,
 'repeatability@1px': 0.0,
 'repeatability@3px': 0.276,
 'repeatability@5px': 0.444}

 Rdnmin