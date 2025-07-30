# code to download the homography dataset

import os
import argparse
import sys
import shutil
import tarfile
import torch
from tqdm import tqdm
from pathlib import Path
import glob

def config():
    parser = argparse.ArgumentParser(description='Download and Manage Revisitop1m dataset')
    parser.add_argument('-i', '--info', action='store_true', help='Get the information of the dataset')
    parser.add_argument('-r', '--remove', action='store_true', help='Remove images from the dataset')
    parser.add_argument('-t', '--target_size', type=int, default=15000, help='Target size of the dataset')
    parser.add_argument('-d', '--download_dir', type=str, default='dataset', help='Directory to save the dataset')
    parser.add_argument('-n', '--num_shards', type=int, default=20, help='Number of shards to download')
    parser.add_argument('-div', '--num_div', type=int, default=None, help='Number of divisions')
    args = parser.parse_args()
    return args

def download_revisitop1m(data_dir, num_files):
    data_dir = Path(data_dir)
    tmp_dir = data_dir.parent / "revisitop1m_tmp"
    if tmp_dir.exists():  # The previous download failed.
        shutil.rmtree(tmp_dir)
    image_dir = tmp_dir / "jpg"
    image_dir.mkdir(exist_ok=True, parents=True)

    url_base = "http://ptak.felk.cvut.cz/revisitop/revisitop1m/"
    list_name = "revisitop1m.txt"

    print("Downloading Revisitop1m list.")
    torch.hub.download_url_to_file(url_base + list_name, tmp_dir / list_name)

    print(f"Downloading Revisitop1m dataset, {num_files} shards.")
    for n in tqdm(range(num_files), position=1):
        tar_name = "revisitop1m.{}.tar.gz".format(n + 1)
        tar_path = image_dir / tar_name
        torch.hub.download_url_to_file(url_base + "jpg/" + tar_name, tar_path)
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=image_dir)
        tar_path.unlink()
    shutil.move(tmp_dir, data_dir)

def get_size(start_path):
    total_size = 0
    print(f"Calculating the size of {start_path}")
    for f in tqdm(start_path.glob("**/*")):
        total_size += f.stat().st_size if f.is_file() else 0
    
    # convert to GB
    total_size = total_size / (1024**3)
    return total_size

def get_download_info(data_dir):
    data_dir = Path(data_dir)
    num_images = len(list(data_dir.glob("**/*.jpg")))
    size = get_size(data_dir)

    print(f"Number of images: {num_images}")
    print(f"Size: {size:.2f} GB")

def remove_images(data_dir, target_size):
    """
    Remove images randomly from the dataset until the number of images is smaller than the target size.
    """
    data_dir = Path(data_dir)
    images = list(data_dir.glob("**/*.jpg"))
        
    num_images = len(images)
    if num_images <= target_size:
        print("The number of images is already smaller than the target size.")
        return
    
    img_to_remove = num_images - target_size
    print(f"Removing {img_to_remove} images to reach from {num_images} to {target_size} images.")

    # randomly select 'img_to_remove' indices
    indices = torch.randperm(num_images)[:img_to_remove]
    for idx in tqdm(indices):
        images[idx].unlink()

def gen_divided_list(data_dir, num_div = 4):
    data_dir = Path(data_dir)
    images = list(data_dir.glob("**/*.jpg"))
    num_images = len(images)
    num_per_div = num_images // num_div

    divided_list = []
    for i in range(num_div):
        start = i * num_per_div
        end = (i + 1) * num_per_div if i != num_div - 1 else num_images
        divided_list.append(images[start:end])
    return divided_list

# Command to launch download: 
# python download_revisitop1m.py --download_dir images --num_shards 1
def main():
    args = config()

    if not args.info:
        download_revisitop1m(args.download_dir, args.num_shards)
    if args.remove:
        remove_images(args.download_dir, args.target_size)
    if args.num_div:
        divided_list = gen_divided_list(args.download_dir, args.num_div)
        fn = ['hardik', 'anna', 'ramu', 'francois']
        for i, images in enumerate(divided_list):
            print(f"Division {i}: {len(images)} images")

            # create a txt file for each division
            with open(f'{args.download_dir}/division_{fn[i]}.txt', 'w') as f:
                for img in images:
                    f.write(f'{img}\n')

    get_download_info(args.download_dir)

if __name__ == '__main__':
    main()