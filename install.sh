
echo "Cloning submodules"
git submodule update --init --recursive

sudo apt-get install cmake libeigen3-dev libceres-dev g++ python3.10-dev


echo "Installing submodules"

packages=(
    "other/DeepLSD"
    "other/faster_pytlsd"
    "other/homography_est"
    "other/dad"
    "other/points_lsd"
    "other/DeDoDe"
    "other/wireframe-detector"
)

for dir in "${packages[@]}"; do
    echo "Installing package in $dir..."
    (cd "$dir" && pip install .) || {
        echo "❌ Failed to install in $dir"
        return 1
    }
done

echo "In case you need to generate the line, please also install afm_op (Remove the exit 0 statement)"
exit 0

# Taken here https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Debian&target_version=12&target_type=deb_network
wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-9

# Add to bashrc
echo -e '\nexport PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}' >> ~/.bashrc && \
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc

cd other/afm_op && python setup.py install && cd ../..
