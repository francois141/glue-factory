
echo "Cloning submodules"
git submodule update --init --recursive

sudo apt-get install cmake libeigen3-dev libceres-dev g++ python3-dev libopencv-dev

pip install setuptools


echo "Installing submodules"

LOG_DIR="install_logs"
mkdir -p "$LOG_DIR"

# package_dir:install_command
packages=(
    "other/DeepLSD:pip install ."
    "other/faster_pytlsd:pip install ."
    "other/homography_est:pip install ."
    "other/dad:pip install ."
    "other/points_lsd:pip install ."
    "other/DeDoDe:pip install ."
    "other/wireframe-detector:pip install ."
    "other/PLNet:pip install ."
    "other/dcvn2:pip install --no-build-isolation ."
)

declare -A install_status

for entry in "${packages[@]}"; do
    dir="${entry%%:*}"
    cmd="${entry#*:}"
    pkg_name=$(basename "$dir")
    log_file="$LOG_DIR/${pkg_name}.log"
    echo "Installing package in $dir... (log: $log_file)"
    if (cd "$dir" && eval "$cmd") > "$log_file" 2>&1; then
        install_status["$dir"]="SUCCESS"
        echo "  -> $dir installed successfully"
    else
        install_status["$dir"]="FAILED"
        echo "  -> $dir FAILED (see $log_file)"
    fi
done

echo ""
echo "=============================="
echo "  Installation Summary"
echo "=============================="
all_ok=true
for dir in "${packages[@]}"; do
    status="${install_status[$dir]}"
    if [ "$status" = "SUCCESS" ]; then
        echo "  [OK]   $dir"
    else
        echo "  [FAIL] $dir  (see $LOG_DIR/$(basename "$dir").log)"
        all_ok=false
    fi
done
echo "=============================="
if $all_ok; then
    echo "All packages installed successfully."
else
    echo "Some packages failed to install. Check logs in $LOG_DIR/"
fi
echo ""

# Install CUDA toolkit if nvcc is not available
if ! command -v nvcc &> /dev/null; then
    echo "CUDA toolkit not found. Installing..."
    # Taken here https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Debian&target_version=12&target_type=deb_network
    wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-12-9

    # Add to bashrc
    echo -e '\nexport PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}' >> ~/.bashrc && \
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
else
    echo "CUDA toolkit already available ($(nvcc --version | grep release)). Skipping install."
fi

echo "In case you need to generate the line, please also install afm_op (Remove the exit 0 statement)"
exit 0

cd other/afm_op && python setup.py install && cd ../..
