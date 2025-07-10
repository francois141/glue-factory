echo "Cloning submodules"
git submodule update --init --recursive

sudo apt-get install cmake libeigen3-dev libceres-dev

echo "Installing submodules"

packages=(
    "other/DeepLSD"
    "other/faster_pytlsd"
    "other/homography_est"
)

for dir in "${packages[@]}"; do
    echo "Installing package in $dir..."
    (cd "$dir" && pip install .) || {
        echo "❌ Failed to install in $dir"
        return 1
    }
done

cd other/afm_op && python setup.py install && cd ../..