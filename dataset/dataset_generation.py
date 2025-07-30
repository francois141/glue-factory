from deeplsd.models.deeplsd_inference import DeepLSD

from PIL import Image
import shutil
import os
import numpy as np
import cv2
import torch
import h5py
import glob
import matplotlib.pyplot as plt
import pickle
from sklearn import preprocessing

def df_and_angle_to_offset(df, angle):
    """ Convert a DF and angle representation back to an offset map. """
    # Calculate x and y components of the offset using angle and magnitude (df)
    offset_x = df * np.sin(angle)
    offset_y = df * np.cos(angle)

    # Stack offset_x and offset_y to create the offset map
    offset = np.stack((offset_x, offset_y), axis=-1)

    return offset

def offset_to_df_and_angle( offset):
    """ Convert an offset map into a DF and angle representation. """
    df = np.linalg.norm(offset, axis=-1)
    angle = np.arctan2(offset[:, :, 0], offset[:, :, 1])
    return df, angle

# Deep LSD Config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
conf = {
    'detect_lines': False,  # Whether to detect lines or only DF/AF
    'line_detection_params': {
        'merge': False,  # Whether to merge close-by lines
        'filtering': True,
        # Whether to filter out lines based on the DF/AF. Use 'strict' to get an even stricter filtering
        'grad_thresh': 3,
        'grad_nfa': True,
        # If True, use the image gradient and the NFA score of LSD to further threshold lines. We recommand using it for easy images, but to turn it off for challenging images (e.g. night, foggy, blurry images)
    }
}

# Load the model
ckpt = '../data/weights/deeplsd_md.tar'
ckpt = torch.load(str(ckpt), map_location=device, weights_only=False)
net = DeepLSD(conf)
net.load_state_dict(ckpt['model'])
net = net.to(device).eval()

def get_line_from_image(path):
    img = cv2.imread(path)[:, :, ::-1]
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if img.shape[0] * img.shape[1] > 1000*800:
        return 0, 0, False

    inputs = {'image': torch.tensor(gray_img, dtype=torch.float, device=device)[None, None] / 255.}

    with torch.no_grad():
        out = net(inputs)

    distances = out['df'].cpu()[0]
    angles = out['line_level'].cpu()[0]

    return distances.numpy(), angles.numpy(), True

cnt = 0

# Command to launch dataset generation
# python dataset_generation.py

dataset_dir = 'revisitop1m/jpg'
image_list = 'revisitop1m/revisitop1m.txt'

if os.path.exists(dataset_dir):
    shutil.rmtree(dataset_dir)
os.makedirs(dataset_dir)

for file_path in glob.glob('images/jpg/**/*.jpg', recursive=True):
    # Get path values
    folder_id = file_path.split('/')[-2]
    name = file_path.split('/')[-1]
    hash = name.split('.')[0]

    # Load base image
    image = np.array(Image.open(file_path))

    # Load lines - inference from deep lsd
    distance_map, angle_map, success = get_line_from_image(file_path)

    # Image too big, ignore this image
    if not success:
        continue

    # Create directory
    cur_dir = f'{dataset_dir}/{folder_id}/{hash}'
    os.makedirs(cur_dir)

    bg_mask = (distance_map > 1.8)

    # Save base image
    Image.fromarray(image).save(f'{cur_dir}/base_image.jpg')

    # Create offset map
    offset_map = df_and_angle_to_offset(distance_map, angle_map)

    # Store values in dictionary
    dict = {
        'min_offset': np.min(offset_map, axis=None),
        'max_offset': np.max(offset_map, axis=None),
        'max_distance': np.max(distance_map, axis=None),
    }

    # Normalize distance map
    distance_map /= np.max(distance_map,axis=None)

    # Normalize angle map
    angle_map /= np.pi

    # Normalize offset map
    offset_map = offset_map - np.min(offset_map, axis=None)
    offset_map /= np.max(offset_map, axis=None)

    # Save distance_map
    Image.fromarray(255*distance_map).convert("L").save(f'{cur_dir}/df.jpg')

    # Save angle_map
    # Image.fromarray(255*angle_map).convert("L").save(f'{cur_dir}/angle.jpg')

    # Save offset map
    # Image.fromarray(255 * offset_map[:,:,0]).convert("L").save(f'{cur_dir}/offset_x.jpg')
    # Image.fromarray(255 * offset_map[:,:,1]).convert("L").save(f'{cur_dir}/offset_y.jpg')

    # Save background mask
    Image.fromarray(255 * bg_mask.astype(np.uint8)).convert("L").save(f'{cur_dir}/bg_mask.jpg')

    with open(f'{cur_dir}/values.pkl', 'wb') as f:
        pickle.dump(dict, f)

    # Add to image list
    with open(image_list, 'a') as f:
        f.write(f'{folder_id}/{hash}.jpg\n')

    cnt += 1

    print(f"Done image : {cnt}")