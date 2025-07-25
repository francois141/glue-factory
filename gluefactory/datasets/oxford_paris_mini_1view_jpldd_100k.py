import logging
import pickle
import random
import shutil
import zipfile
from pathlib import Path

import numpy as np
import torch

from gluefactory.datasets import BaseDataset, augmentations
from gluefactory.settings import DATA_PATH, root
from gluefactory.utils.image import ImagePreprocessor, load_image, read_image, numpy_image_to_torch

logger = logging.getLogger(__name__)


class OxfordParisMiniOneViewJPLDD(BaseDataset):
    """
    Subset of the Oxford Paris dataset as defined here: https://cmp.felk.cvut.cz/revisitop/
    Supports loading groundtruth and only serves images for that gt exists.
    Dataset only deals with loading one element. Batching is done by Dataloader!

    Adapted to use POLD2 structure of files -> Files and gt in same folder besides each other
    Some facts:
    - Pold2 gt is generated same size as original image and can be resized
    """

    default_conf = {
        "data_dir": "oxparis_100k",  # as subdirectory of DATA_PATH(defined in settings.py)
        "grayscale": False,
        "train_batch_size": 2,  # prefix must match split
        "test_batch_size": 1,
        "val_batch_size": 1,
        "all_batch_size": 1,
        "device": None,  # specify device to move image data to. if None is given, just read, skip move to device
        "split": "train",  # train, val, test
        "seed": 0,
        "num_workers": 0,  # number of workers used by the Dataloader
        "prefetch_factor": None,
        "reshape": None,  # ex 800  # if reshape is activated AND multiscale learning is activated -> reshape has prevalence
        "square_pad": True,  # square padding is needed to batch together images with current scaling technique(keeping aspect ratio). Can and should be deactivated on benchmarks
        "multiscale_learning": {
            "do": False,
            "scales_list": [
                1000,
                800,
                600,
                400,
            ],  # use interger scales to have resize keep aspect ratio -> not squashing img by forcing it to square
            "scale_selection": "random",  # random or round-robin
        },
        "load_features": {
            "do": False,
            "check_exists": True,
            "point_gt": {
                "data_keys": [
                    "superpoint_heatmap",
                    "gt_keypoints",
                    "gt_keypoints_scores",
                ],
                "load_points": False,
                "use_score_heatmap": False,
                "max_num_heatmap_keypoints": -1,  # topk keypoints used to create the heatmap (-1 = all are used)
                "max_num_keypoints": 63,  # topk keypoints returned as gt keypoint locations (-1 - return all)
                # -> Can also be set to None to return all points but this can only be used when batchsize=1. Min num kp in oxparis: 63
                "use_deeplsd_lineendpoints_as_kp_gt": False,  # set true to use deep-lsd line endpoints as keypoint groundtruth
                "use_superpoint_kp_gt": True,  # set true to use default HA-Superpoint groundtruth
            },
            "line_gt": {
                "data_keys": ["deeplsd_distance_field", "deeplsd_angle_field"],
                "enforce_threshold": 5.0,  # Enforce values in distance field to be no greater than this value
            },
            "augment": {  # there is the option to use data augmentation. It is not enlarging dataset but applies the augmentation to an Image with certain probability
                "do": False,
                "type": "dark",  # choose "identity" for no augmentation; other options are "lg", "dark"
            },
        },
        "img_list": "gluefactory/datasets/oxford_paris_images_100k.txt",
        # img list path from repo root -> use checked in file list, it is similar to pold2 file
        "rand_shuffle_seed": None,  # seed to randomly shuffle before split in train and val
        "val_size": 2000,  # size of validation set given
        "train_size": 100000,
    }

    def _init(self, conf):

        data_directory = DATA_PATH / conf.data_dir

        # Auto-download the dataset if not existing
        if not data_directory.exists():
            assert False, "We should not arrive here"
            self.download_oxford_paris_mini_100k()

        # Build the output path
        output_path = root / self.conf.img_list

        if not output_path.exists():
            # Recursively list all files containing 'base_image' in their path
            matching_files = []

            for img_file in data_directory.rglob("*base_image.jpg"):
                stem = img_file.stem.replace("base_image", "")  # get the base prefix
                base_dir = img_file.parent

                # Build expected paths
                df_file = base_dir / f"{stem}df.npy"
                af_file = base_dir / f"{stem}af.npy"
                heatmap_file = base_dir / f"{stem}heatmap.npy"

                # Check all 4 files exist and are non-empty
                files = [img_file, df_file, af_file, heatmap_file]
                if all(p.exists() and p.stat().st_size > 0 for p in files):
                    matching_files.append(img_file)
                else:
                    print(f"Skipping {img_file} — one or more files missing or empty.")


            # Ensure the parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write each path to a new line in the output file
            with open(output_path, "w") as f:
                for file_path in matching_files:
                    f.write(str(file_path) + "\n")
                
        # Log how many images we have
        print(f"Dataset contains {len(matching_files)} images")

        with open(root / self.conf.img_list, "r") as f:
            self.img_list = [file_name.strip("\n") for file_name in f.readlines()]

        # load image names
        images = self.img_list
        if self.conf.rand_shuffle_seed is not None:
            np.random.RandomState(conf.shuffle_seed).shuffle(images)
        train_images = images[: conf.train_size]
        val_images = images[conf.train_size : conf.train_size + conf.val_size]
        self.images = {
            "train": train_images,
            "val": val_images,
            "test": images,
            "all": images,
        }
        logger.info(f"DATASET OVERALL(NO-SPLIT) IMAGES: {len(images)}")

        augmentation_map = {
            "dark": augmentations.DarkAugmentation,
            "lg": augmentations.LGAugmentation,
            "identity": augmentations.IdentityAugmentation,
        }

        self.augmentation = augmentation_map[self.conf.load_features.augment.type]()

    def download_oxford_paris_mini_100k(self):
        """
        The downloaded dataset already contains ground-truth keypoints, line-df and line-af and lines
        for dataset original resolution.
        """
        logger.info("Downloading the 100k OxfordParis Mini dataset...")
        oxparis_root_directory = (DATA_PATH / self.conf.data_dir).parent
        tmp_dir = oxparis_root_directory.parent / "oxpa_100k_tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(exist_ok=True, parents=True)
        url = "TODO: Link"
        zip_name = "oxparis.zip"
        zip_path = tmp_dir / zip_name
        torch.hub.download_url_to_file(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmp_dir)
        shutil.move(tmp_dir / "oxparis_100k", str(oxparis_root_directory))
        logger.info("Delete temporary files...")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    def get_dataset(self, split):
        assert split in ["train", "val", "test", "all"]
        return _Dataset(self.conf, self.images[split], split, self.augmentation)


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, image_sub_paths: list[str], split, augmentation):
        super().__init__()
        self.split = split
        self.conf = conf
        self.augmentation = augmentation
        self.grayscale = bool(conf.grayscale)
        self.max_num_gt_kp = conf.load_features.point_gt.max_num_keypoints

        self.use_superpoint_kp_gt = conf.load_features.point_gt.use_superpoint_kp_gt
        self.use_dlsd_ep_as_kp_gt = (
            conf.load_features.point_gt.use_deeplsd_lineendpoints_as_kp_gt
        )

        # Initialize Image Preprocessors for square padding and resizing
        self.preprocessors = {}  # stores preprocessor for each reshape size
        if self.conf.reshape is not None:
            self.register_image_preprocessor_for_size(self.conf.reshape)
        if self.conf.multiscale_learning.do:
            for scale in self.conf.multiscale_learning.scales_list:
                self.register_image_preprocessor_for_size(scale)

        if self.conf.multiscale_learning.do:
            if self.conf.multiscale_learning.scale_selection == "round-robin":
                self.scale_selection_idx = 0
            # Keep track uf how many selected with current scale for batching (all img in same batch need same size)
            self.num_select_with_current_scale = 0
            self.current_scale = None
        # we need to make sure that the appropriate batch size for the dataset conf is set correctly.
        self.relevant_batch_size = self.conf[f"{split}_batch_size"]

        self.img_dir = DATA_PATH / conf.data_dir
        self.dlsd_kp_gt_folder = self.img_dir.parent / "deeplsd_kp_gt"

        # Extract image paths
        self.image_sub_paths = image_sub_paths  # [Path(i) for i in image_sub_paths]

        # making them relative for system independent names in export files (path used as name in export)
        if len(self.image_sub_paths) == 0:
            raise ValueError(f"Could not find any image in folder: {self.img_dir}.")
        logger.info(f"NUMBER OF IMAGES: {len(self.image_sub_paths)}")
        logger.info(
            f"KNOWN BATCHSIZE FOR MY SPLIT({self.split}) is {self.relevant_batch_size}"
        )
        # Load features
        if conf.load_features.do:
            # filter out where missing groundtruth
            new_img_path_list = []
            for img_path in self.image_sub_paths:
                if not self.conf.load_features.check_exists:
                    new_img_path_list.append(img_path)
                    continue
                # perform checks
                full_artificial_img_path = Path(self.img_dir / img_path)
                img_folder = (
                    full_artificial_img_path.parent / full_artificial_img_path.stem
                )
                keypoint_file = img_folder / "keypoint_scores.npy"
                af_file = img_folder / "angle.jpg"
                df_file = img_folder / "df.jpg"

                dlsd_kp_subfolder = (
                    self.dlsd_kp_gt_folder / full_artificial_img_path.stem
                )
                dlsd_kp_file = dlsd_kp_subfolder / "base_image.npy"

                if keypoint_file.exists() and af_file.exists() and df_file.exists():
                    if self.use_dlsd_ep_as_kp_gt and dlsd_kp_file.exists():
                        new_img_path_list.append(img_path)
                        continue
                    new_img_path_list.append(img_path)

            self.image_sub_paths = new_img_path_list
            logger.info(f"NUMBER OF IMAGES WITH GT: {len(self.image_sub_paths)}")

        self.img_dir = DATA_PATH / conf.data_dir
        self.image_sub_paths = [
            Path(str(p)[:-15]) for p in image_sub_paths
        ]  

    def register_image_preprocessor_for_size(self, size: int) -> None:
        """
        We use image preprocessor to reshape images and square pad them. We resize keeping the aspect ratio of images.
        Thus image sizes can be different even when long side scaled to same length. Thus square padding is needed so that
        all images can be stuck together in a batch.
        """
        self.preprocessors[size] = ImagePreprocessor(
            {
                "resize": size,
                "edge_divisible_by": None,
                "side": "long",
                "interpolation": "bilinear",
                "align_corners": None,
                "antialias": True,
                "square_pad": bool(self.conf.square_pad),
                "add_padding_mask": True,
            }
        )

    def _read_base_image(self, img_folder_path, enforce_batch_dim=False):
        pass

    def _read_df(self, img_folder_path, enforce_batch_dim=False):
        """
        Read distance field as tensor and puts it on device
        """
        img_path = str(img_folder_path) + "_df.npy"
        img = np.load(img_path)

        img = torch.tensor(img)

        if self.conf.device is not None:
            img = img.to(self.conf.device)
        return img
        
    def _read_af(self, img_folder_path, enforce_batch_dim=False):
        """
        Read distance field as tensor and puts it on device
        """
        img_path = str(img_folder_path) + "_af.npy"
        img = np.load(img_path)

        img = torch.tensor(img)

        if self.conf.device is not None:
            img = img.to(self.conf.device)
        return img

    def _read_keypoint_mask(self, img_folder_path, enforce_batch_dim=False):
        """
        Read distance field as tensor and puts it on device
        """
        img_path = str(img_folder_path) + "_heatmap.npy"
        img = np.load(img_path)

        img = torch.tensor(img)

        if self.conf.device is not None:
            img = img.to(self.conf.device)
        return img

    def _read_image(self, img_folder_path, enforce_batch_dim=False):
        """
        Read image as tensor and puts it on device
        """
        img_path = str(img_folder_path) + "_base_image.jpg"
        img = load_image(img_path, grayscale=self.grayscale)
        if enforce_batch_dim:
            if img.ndim < 4:
                img = img.unsqueeze(0)
        assert img.ndim >= 3
        if self.conf.device is not None:
            img = img.to(self.conf.device)
        return img

    def _read_groundtruth(self, image_folder_path, shape: int = None) -> dict:
        """
        Reads groundtruth for points and lines from respective files
        We can assume that gt files are existing at this point->checked in init!

        DeepLSD line endpoints can be loaded and used as keypoint groundtruth. This can be configured in the dataset config.
        One can also choose to only use deepLSD line endpoints.

        image_folder_path: full image folder path to get the gt data from
        original_img_size: format w,h original image size to be able to create heatmaps.
        shape: shape to reshape img to if it is not None
        """
        # Load config and local variables
        reshape_scales = None
        reshaped_size = None
        heatmap_gt_key_name = self.conf.load_features.point_gt.data_keys[0]
        kp_gt_key_name = self.conf.load_features.point_gt.data_keys[1]
        kp_score_gt_key_name = self.conf.load_features.point_gt.data_keys[2]
        df_gt_key = self.conf.load_features.line_gt.data_keys[0]
        af_gt_key = self.conf.load_features.line_gt.data_keys[1]

        features = {}
        # load keypoint gt file paths
        kp_file = image_folder_path / "keypoints.npy"
        kps_file = image_folder_path / "keypoint_scores.npy"
        dlsd_kp_gt_file = (
            self.dlsd_kp_gt_folder / image_folder_path.name / "base_image.npy"
        )

        # Load distance field
        distance_field = self._read_df(image_folder_path)
        thres = self.conf.load_features.line_gt.enforce_threshold
        if thres is not None:
            distance_field = torch.where(distance_field > thres, thres, distance_field)

        # Load angle field
        angle_field = self._read_af(image_folder_path)

        # Load keypoint heatmap
        keypoint_scores = self._read_keypoint_mask(image_folder_path)

        # Reshape maps if needed
        if shape is not None:
            preprocessor_df_out = self.preprocessors[shape](distance_field.unsqueeze(0))
            reshape_scales = preprocessor_df_out["scales"]
            reshaped_size = preprocessor_df_out["image_size"]
            distance_field = preprocessor_df_out["image"].squeeze(
                0
            )  # only store image here as padding map will be stored by preprocessing image
            angle_field = self.preprocessors[shape](angle_field.unsqueeze(0))["image"].squeeze(0)
            keypoint_scores = self.preprocessors[shape](keypoint_scores.unsqueeze(0))["image"].squeeze(0)

        features[df_gt_key] = distance_field
        features[af_gt_key] = angle_field
        features[heatmap_gt_key_name] = keypoint_scores

        return features
    
    def __getitem__(self, idx):
        """
        Dataloader is usually just returning one datapoint by design. Batching is done in Loader normally.
        """
        full_artificial_img_path = self.img_dir / self.image_sub_paths[idx]
        folder_path = full_artificial_img_path.parent / full_artificial_img_path.stem
        img = self._read_image(folder_path)

        if self.conf.load_features.augment.do:
            print("Check this holds as well for this dataset")
            raise NotImplemented
            try:
                img = img.numpy().transpose(1, 2, 0)
                img = self.augmentation(image=img, return_tensor=True)
            except Exception as e:
                logging.error(f"Error in augmentation: {e}")
        orig_shape = img.shape[-1], img.shape[-2]
        size_to_reshape_to = self.select_resize_shape(orig_shape)
        data = {
            "name": str(folder_path / "base_image.jpg"),
        }  # keys: 'name', 'scales', 'image_size', 'transform', 'original_image_size', 'image'
        if size_to_reshape_to == orig_shape:
            data["image"] = img
        else:
            data = {**data, **self.preprocessors[size_to_reshape_to](img)}
        if self.conf.load_features.do:
            gt = self._read_groundtruth(
                folder_path,
                None if size_to_reshape_to == orig_shape else size_to_reshape_to,
            )
            data = {**data, **gt}
        return data

    def do_change_size_now(self) -> bool:
        """
        Based on current state decides whether to change shape to reshape images to.
        This decision is needed as all images in a batch need same shape. So we only potentially change shape
        when a new batch is starting.

        Returns:
            bool: should shape be potentially changed?
        """
        # check if batch changes
        if self.num_select_with_current_scale % self.relevant_batch_size == 0:
            self.num_select_with_current_scale = 0  # if batch changes set counter to 0
            return True
        else:
            return False

    def select_resize_shape(self, original_img_size: tuple):
        """
        Depending on whether resize or multiscale learning is activated the shape to resize the
        image to is returned. If none of it is activated, the original image size will be returned.
        Reshape has prevalence over multiscale learning!
        """
        do_reshape = self.conf.reshape is not None
        do_ms_learning = self.conf.multiscale_learning.do
        if not do_reshape and not do_ms_learning:
            return original_img_size

        if do_reshape:
            return int(self.conf.reshape)

        if do_ms_learning:
            if self.do_change_size_now():
                self.num_select_with_current_scale += 1
                scales_list = self.conf.multiscale_learning.scales_list
                scale_selection = self.conf.multiscale_learning.scale_selection
                assert (
                    len(scales_list) > 1
                )  # need more than one scale for multiscale learning to make sense

                if scale_selection == "random":
                    choice = int(random.choice(scales_list))
                    self.current_scale = choice
                    return choice
                elif scale_selection == "round-robin":
                    current_scale = scales_list[self.scale_selection_idx]
                    self.current_scale = current_scale
                    self.scale_selection_idx += 1
                    self.scale_selection_idx = self.scale_selection_idx % len(
                        scales_list
                    )
                    return int(current_scale)
            else:
                self.num_select_with_current_scale += 1
                return self.current_scale

        raise Exception("Shouldn't end up here!")

    def set_num_selected_with_current_scale(self, value: int) -> None:
        """
        Sets the self.num_selected_with_current_scale variable to a certain value.
        This method is implemented as interface for the MergedDataset to be able to deal with multiscale learning
        on multiple datasets

        Args:
            value (int): new value for variable
        """
        self.num_select_with_current_scale = value

    def get_current_scale(self) -> int:
        """
        Returns the current used scale to reshape images to. Returns None if multiscale learning is deactivated.
        This method is implemented as interface for the MergedDataset to be able to deal with multiscale learning
        on multiple datasets

        Returns:
            int: current scale used to reshape in multi-scale training. None if its deactivated
        """
        return self.current_scale

    def set_current_scale(self, value):
        """
        Sets the current scale used for multiscale training. Used to set size of reshape of this dataset during batch.
        This method is implemented as interface for the MergedDataset to be able to deal with multiscale learning
        on multiple datasets.

        Returns:
            int: current scale used to reshape in multi-scale training. None if its deactivated
        """
        self.current_scale = value

    def __len__(self):
        return len(self.image_sub_paths)
