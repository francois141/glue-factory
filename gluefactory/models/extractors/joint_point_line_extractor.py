import logging

import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.transform import warp_perspective

from gluefactory.geometry.kp_losses import soft_argmax_only_loss
from gluefactory.models import get_model
from gluefactory.models.backbones.backbone_encoder import AlikedEncoder, aliked_cfgs
from gluefactory.models.backbones.vit_encoder import VITBackbone
from gluefactory.models.base_model import BaseModel
from gluefactory.models.extractors.aliked import DKD, SDDH, SMH, InputPadder
from gluefactory.models.extractors.dad import DadDetector
from gluefactory.models.extractors.dedode import DeDoDeDetector
from gluefactory.models.extractors.superpoint_open import SuperPoint
from gluefactory.models.lines.fast_lsd_extractor import FastLSDLineExtractor
from gluefactory.models.extractors.dad_distill import DadDistillDetector
from gluefactory.utils.misc import change_dict_key
from gluefactory.models.extractors.joint_point_line_extractor_utils import *

aliked_checkpoint_url = "https://github.com/Shiaoming/ALIKED/raw/main/models/{}.pth"  # used for training based on ALIKED weights
logger = logging.getLogger(__file__)

class JointPointLineDetectorDescriptor(BaseModel):
    """
    Pipeline to jointly detect and describe keypoints and lines given an RGB image.

    If a checkpoint is loaded, the config from the checkpoint is not. We could change that in the future.
    """

    # default checkpoint used for automatic weight loading if no other path specified
    # currently its the oxparis-800-focal checkpoint
    jpl_default_checkpoint_url = (
        "https://polybox.ethz.ch/index.php/s/IN0dxL4ljUacf9K/download"
    )

    default_conf = {
        "backbone": "aliked",  # backbone encoder to use, options: aliked - vit
        "aliked_model_name": "aliked-n16",  # ALIKED model determining architecture of our backbone
        "line_df_decoder_channels": 32,
        "max_num_keypoints": 1024,  # setting for training, for eval: -1
        "detection_threshold": -1,  # setting for training, for eval: 0.2
        "nms_radius": 3,
        "subpixel_refinement": True,  # perform subpixel refinement after detection
        "force_num_keypoints": False,
        "freeze_lines": False,
        "descriptor_branch": "aliked", # options are aliked or dedode
        "training": {  # training settings
            "do": False,  # switch to turn off other settings regarding training = "training mode"
            "two_view": False,  # whether training is done with a two-view pipeline (True) or with a one-view pipeline (False)
            "aliked_pretrained": True,  # use pretrained ALIKED weights in backbone encoder
            "pretrain_kp_decoder": True,  # use pretrained ALIKED weights for keypoint-heatmap decoder
            "train_descriptors": {  # for train descriptors in one-view: generate gt descriptors, other losses for two_view
                "gt_aliked_model": "aliked-n32", # dedode is also an option
                "use_one_view_loss": True,  # In one view training can decide if train descriptors with this flag
                "use_two_view_loss": True,  # can only be used if two_view training activated (sparseNRE loss)
            },
            "loss": {
                "use_one_view_df_loss": True,  # one-view losses can be applied any time
                "use_two_view_df_loss": True,  # two view losses are only applied if two-view training activated
                "use_one_view_kp_loss": True,  # use one-view keypoint loss ex. focal, l1 etc
                "kp_loss_name": "focal_loss",  # other options: bce, weighted_bce or focal loss
                "kp_loss_parameters": {
                    "lambda_weighted_bce": 200,  # weighted bce parameter factor how to boost keypoint loss in map
                    "focal_gamma": 2,
                    # focal loss parameter controlling how strong to focus on hard examples (typical range 1-5)
                    "focal_alpha": 0.25,  # focal loss parameter to mitigate class imbalances
                },
                "refinement_radius": 5,  # radius for softargmax loss
                "loss_weights": {
                    "one_view_line_df_weight": 1,
                    "two_view_line_df_weight": 1,
                    "keypoint_weight": 1,
                    "one_view_descriptor_weight": 1,
                    "two_view_descriptor_weight": 1,
                    "softargmax_weight": 1,  # if > 0 activates calculation of soft argmax loss on keypoint detection. Only used if two_view activated
                },
            },
        },
        "line_detection": {  # by default we use the POLD2 Line Extractor (MLP with Angle Field)
            "do": True,
            "name": "lines.fast_lsd_extractor",
            "conf": FastLSDLineExtractor.default_conf,
        },
        "checkpoint": jpl_default_checkpoint_url,  # if given and non-null, load model checkpoint if local path load locally if standard url download it.
        "line_neighborhood": 5,  # used to normalize / denormalize line distance field
        "trainable": True,
    }

    n_limit_max = 20000  # taken from ALIKED which gives max num keypoints to detect!

    required_data_keys = ["image"]

    def _init(self, conf) -> None:
        logger.debug(f"final config dict(type={type(conf)}): {conf}")
        # set loss fn
        assert self.conf.training.loss.kp_loss_name in [
            "weighted_bce",
            "focal_loss",
            "bce",
            "distill",
            "distill_dad",
            "distill_dad_superpoint_focal",
            "distill_dad_superpoint_bce",
            "distill_dad_superpoint_bce_weighted",
        ]
        if self.conf.training.loss.kp_loss_name == "weighted_bce":
            logger.warning("-- Using weighted BCE loss for the points! --")
            self.loss_fn = self.weighted_bce_loss
        elif self.conf.training.loss.kp_loss_name == "focal_loss":
            logger.warning("-- Using focal loss for the points! --")
            self.loss_fn = self.focal_loss
        elif self.conf.training.loss.kp_loss_name == "distill_dad_superpoint_focal":
            logger.warning("-- Using focal loss for the points! --")
            self.loss_fn = self.focal_loss
        elif self.conf.training.loss.kp_loss_name == "distill_dad_superpoint_bce":
            logger.warning("-- Using BCE loss for the points! --")
            self.loss_fn = nn.BCELoss(reduction="none")
        elif self.conf.training.loss.kp_loss_name == "distill_dad_superpoint_bce_weighted":
            logger.warning("-- Using weighted BCE loss for the points! --")
            self.loss_fn = self.weighted_bce_loss
        else:
            logger.warning("-- Using BCE loss for the points! --")
            self.loss_fn = nn.BCELoss(reduction="none")
        # c1-c4 -> output dimensions of encoder blocks, dim -> dimension of hidden feature map
        # K=Kernel-Size, M=num sampling pos
        aliked_model_cfg = aliked_cfgs[conf.aliked_model_name]
        dim = aliked_model_cfg["dim"]
        K = aliked_model_cfg["K"]
        M = aliked_model_cfg["M"]
        self.lambda_valid_kp = conf.training.loss.kp_loss_parameters.lambda_weighted_bce
        # Load Network Components
        logger.info(f"Using {self.conf.backbone} backbone")
        if self.conf.backbone == "vit":
            self.encoder_backbone = VITBackbone()
        elif self.conf.backbone == "aliked":
            self.encoder_backbone = AlikedEncoder(
                aliked_model_cfg
            )
        else:
            print("Unknown backbone")
            raise NotImplementedError

        self.keypoint_and_junction_branch = SMH(dim)  # using SMH from ALIKE here

        self.dkd = DKD(  # heuristic point-detection with subpixel refinement from ALIKE (remove border points, nms, refinement)
            radius=conf.nms_radius,
            top_k=-1 if conf.detection_threshold > 0 else conf.max_num_keypoints,
            scores_th=conf.detection_threshold,
            n_limit=(
                conf.max_num_keypoints
                if conf.max_num_keypoints > 0
                else self.n_limit_max
            ),
        )
        # Keypoint descriptor module "SDDH" from ALIKED
        if self.conf.descriptor_branch == "aliked":
            self.descriptor_branch = SDDH(
                dim, K, M, gate=nn.SELU(inplace=True), conv2D=False, mask=False
            )
        else:
            self.descriptor_branch = DeDoDeDetector({})

        # Line distance field decoder similar to that in DeepLSD
        self.distance_field_branch = nn.Sequential(
            nn.Conv2d(dim, conf.line_df_decoder_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(conf.line_df_decoder_channels),
            nn.Conv2d(
                conf.line_df_decoder_channels,
                conf.line_df_decoder_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(conf.line_df_decoder_channels),
            nn.Conv2d(conf.line_df_decoder_channels, 1, kernel_size=1),
            nn.ReLU(),
        )

        # load pretrained_elements if wanted (for now that only the ALIKED parts of the network)
        if conf.training.do and conf.training.aliked_pretrained:
            logger.warning("Load pretrained weights for aliked parts...")
            self.load_pretrained_aliked_elements()

        # Initialize Lightweight ALIKED model to perform on-the-fly ground-truth generation for descriptors if training in one-view setting
        if conf.training.do and conf.training.train_descriptors.use_one_view_loss:
            if "aliked" not in self.conf.training.train_descriptors.gt_aliked_model:
                logger.warning("Load DeDoDe model for descriptor training...")
                self.descriptor_gt = DeDoDeDetector({})
            else:
                logger.warning("Load ALiked Lightweight model for descriptor training...")
                aliked_gt_cfg = {
                    "model_name": self.conf.training.train_descriptors.gt_aliked_model,
                    "max_num_keypoints": self.conf.max_num_keypoints,
                    "detection_threshold": self.conf.detection_threshold,
                    "force_num_keypoints": False,
                    "pretrained": True,
                    "nms_radius": self.conf.nms_radius,
                }
                self.aliked_lw = get_model("extractors.aliked_light")(aliked_gt_cfg).eval()

        # load model checkpoint if given -> only load weights
        if conf.checkpoint is not None:
            if Path(conf.checkpoint).exists():
                logger.warning(
                    f"Load model parameters from local checkpoint {conf.checkpoint}"
                )
                chkpt_statedict = torch.load(
                    conf.checkpoint, map_location=torch.device("cpu")
                )
            else:
                logger.warning(
                    f"Try Load model parameters from URL checkpoint {conf.checkpoint}"
                )
                chkpt_statedict = torch.hub.load_state_dict_from_url(
                    conf.checkpoint, map_location="cpu"
                )

            # Extract from two-view
            chkpt_statedict["model"] = {
                k.split("extractor.")[-1]: v
                for k, v in chkpt_statedict["model"].items()
            }

            # remove mlp weights from line detection
            chkpt_statedict["model"] = {
                k: v for k, v in chkpt_statedict["model"].items() if not ("mlp" in k)
            }

            self.load_state_dict(
                chkpt_statedict["model"], strict=False
            )  # set to True to check if all keys are present (mlp weights are not present as we removed them above)
        elif conf.checkpoint is not None:
            chkpt_statedict = torch.hub.load_state_dict_from_url(
                conf.checkpoint, map_location=torch.device("cpu")
            )
            # Extract from two-view
            chkpt_statedict["model"] = {
                k.split("extractor.")[-1]: v
                for k, v in chkpt_statedict["model"].items()
            }
            self.load_state_dict(chkpt_statedict["model"], strict=False)

        logger.info(f"Load line extractor: {self.conf.line_detection.name}")
        self.line_extractor = get_model(self.conf.line_detection.name)(
            self.conf.line_detection.conf
        )

        if self.conf.training.loss.kp_loss_name == "distill":
            # Use dad distillation
            self.dad_distil = DadDistillDetector({
                "max_num_keypoints": 1024,
            })

        # Freeze df and backbone extractor
        if self.conf.freeze_lines:
            logger.warning("Freezing line extractor and distance field branch!")

            for param in self.encoder_backbone.parameters():
                param.requires_grad = False
            for param in self.distance_field_branch.parameters():
                param.requires_grad = False

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dad_model = DadDetector({
            "max_num_keypoints": None,
            "nms_radius": 4,
            "detection_threshold": 0.005,
            "remove_borders": 4,
            "descriptor_dim": 256,
            "channels": [64, 64, 128, 128, 256],
            "dense_outputs": None,
            "weights": None,  # local path of pretrained weights
        }).to(device)
        self.dad_model.eval().to(device)

        self.superpoint_model = SuperPoint({}).to(device)
        self.superpoint_model.eval().to(device)

    def _forward(self, data: dict) -> torch.Tensor:
        """
        Perform a forward pass. Certain things are only executed NOT in training mode.
        Returned:
            - Probabilistic Keypoint Heatmap
            - Detected Keypoints
            - Keypoint descriptors (sparse, do one for every detected keypoint)
            - DeepLSD like Distance field (denormalized)
            - DeepLSD like Angle Field (between -Pi and Pi as radians)
            - Detected Lines (if line detection activated)
        """
        # output container definition
        output = {}

        # pad image
        image = data["image"]
        div_by = 2**5
        padder = InputPadder(image.shape[-2], image.shape[-1], div_by)

        # Get Hidden Feature Map and Keypoint/junction scoring
        padded_img = padder.pad(image)

        # pass through encoder
        if self.conf.freeze_lines:
            self.encoder_backbone.eval()
            with torch.no_grad():
                feature_map_padded = self.encoder_backbone(padded_img)
        else:
            feature_map_padded = self.encoder_backbone(padded_img)

        # pass through keypoint & junction decoder
        score_map_padded = self.keypoint_and_junction_branch(feature_map_padded)

        # normalize and remove padding and format dimensions
        feature_map_padded_normalized = torch.nn.functional.normalize(
            feature_map_padded, p=2, dim=1
        )
        feature_map = padder.unpad(feature_map_padded_normalized)
        logger.debug(
            f"Image size: {image.shape}\nFeatureMap-unpadded: {feature_map.shape}\nFeatureMap-padded: {feature_map_padded.shape}"
        )
        keypoint_and_junction_score_map = padder.unpad(
            score_map_padded
        )  # B x 1 x H x W

        # Used to visualise the intermediate backbone using PCA
        output["backbone"] = feature_map

        # For storing, remove additional dimension but keep batch dimension even if its 1
        # but keep additional dimension for variable -> needed by dkd
        if keypoint_and_junction_score_map.shape[0] == 1:
            output["keypoint_and_junction_score_map"] = keypoint_and_junction_score_map[
                :, 0, :, :
            ]  # B x H x W
        else:
            output["keypoint_and_junction_score_map"] = (
                keypoint_and_junction_score_map.squeeze()
            )  # B x H x W

        ## Line DF Decoder ##
        if self.conf.freeze_lines:
            self.distance_field_branch.eval()
            with torch.no_grad():
                line_distance_field = self.denormalize_df(
                    self.distance_field_branch(feature_map)
                )  # denormalize as NN outputs normalized version
        else:
            line_distance_field = self.denormalize_df(
                self.distance_field_branch(feature_map)
            )  # denormalize as NN outputs normalized version
        # remove additional dimensions of size 1 if not having batchsize one
        line_distance_field = (
            line_distance_field.squeeze(1)
            if line_distance_field.shape[0] == 1
            else line_distance_field.squeeze()
        )
        output["line_distancefield"] = line_distance_field

        # Keypoint detection also removes kp at border. it can return topk keypoints or set of thresholded kp.
        keypoints, _, kptscores = self.dkd(
            keypoint_and_junction_score_map,
            sub_pixel=bool(self.conf.subpixel_refinement),
        )

        # raw output of DKD needed to generate GT-Descriptors (ONLY done if one-view-loss used)
        if (
            self.conf.training.do
            and self.conf.training.train_descriptors.use_one_view_loss
        ):
            output["keypoints_raw"] = keypoints

        _, _, h, w = image.shape
        wh = torch.tensor([w, h], device=image.device)
        # no padding required, can set detection_threshold=-1 and conf.max_num_keypoints -> HERE WE SET THESE VALUES
        # SO WE CAN EXPECT SAME NUM!
        rescaled_kp = wh * (torch.stack(keypoints) + 1.0) / 2.0
        output["keypoints"] = rescaled_kp
        output["keypoint_scores"] = torch.stack(kptscores)

        # Keypoint descriptors
        if self.conf.descriptor_branch == "aliked":
            keypoint_descriptors, _ = self.descriptor_branch(feature_map, keypoints)
            output["descriptors"] = torch.stack(keypoint_descriptors)  # B N D
        else:
            output["descriptors"] = self.descriptor_branch.describe_keypoints({"image": data["image"]}, keypoints[0].unsqueeze(0))

        ## Line Detection ##
        # Only Perform line detection when NOT in training mode
        if self.conf.line_detection.do and not self.training:
            if output.get("line_anglefield", None) is None:
                # create dummy so that zipping works
                line_angle_field = torch.zeros_like(line_distance_field)

            # Perform forward pass for line detector, batching handled internally
            line_data = {
                "line_anglefield": line_angle_field,
                "line_distancefield": line_distance_field,
                "image": image,
                "keypoints": rescaled_kp,
                "kp_descriptors": output["descriptors"],
            }
            pred_line_data = self.line_extractor(line_data)

            output["lines"] = torch.stack(pred_line_data["lines"], dim=0)
            if self.conf.line_detection.conf.return_line_descriptors:
                output["line_descriptors"] = torch.stack(
                    pred_line_data["line_descriptors"], dim=0
                )
            output["valid_lines"] = torch.stack(pred_line_data["valid_lines"], dim=0)

        return output


    def loss(self, pred: dict, data: dict) -> dict:
        """
        format of data: B x H x W
        perform loss calculation based on prediction and data(=groundtruth) for a batch.
        If predictions contain padding_mask we consider this on loss calculation
        1. On Keypoint-ScoreMap:        weighted BCE Loss / BCE Loss / Focal Loss
        2. On Keypoint-Descriptors:     L1 loss
        3. On Line-Distance Field:      use L1 loss on normalized versions of Distance field (as in deepLSD paper)
        """

        losses = {}
        metrics = {}
        losses["total"] = 0

        prediction_dict = {}
        if self.conf.training.two_view:
            for k, v in pred.items():
                if k.endswith("0"):
                    prediction_dict[k[:-1]] = v
                else:
                    prediction_dict[k] = v
        else:
            prediction_dict = pred

        # Load view0 ground truth if two view
        gt_dict_view0 = data["view0"]["cache"] if self.conf.training.two_view else data
        gt_dict_view1 = data["view1"]["cache"] if self.conf.training.two_view else None
        H = (
            data["H_0to1"] if self.conf.training.two_view else None
        )  # for each sample in batch there is a separate homography
        H_inv = (
            torch.linalg.inv(H) if self.conf.training.two_view else None
        )  # inv calc supports batch of matrix

        img = data["view0"]["image"] if self.conf.training.two_view else data["image"]
        # define padding mask which is only ones if no padding is used -> makes loss compatible with any scaling technique and whether padding is used or not
        padding_mask_view0 = gt_dict_view0.get(
            "padding_mask", torch.ones_like(img)
        )[  # TODO: padding is not the same
            :, 0, :, :
        ].int()
        df_gt_mask_view0 = (
            gt_dict_view0["deeplsd_distance_field"] < self.conf.line_neighborhood
        )
        df_gt_mask_view1 = (
            gt_dict_view1["deeplsd_distance_field"] < self.conf.line_neighborhood
            if self.conf.training.two_view
            else None
        )
        
        # Distance field loss. Depends on the pipeline (two-view or one-view)
        # use normalized versions for loss
        if self.conf.training.loss.use_one_view_df_loss:
            line_df = prediction_dict["line_distancefield"]
            deeplsd_line_df =  gt_dict_view0["deeplsd_distance_field"]

            losses["one_view_line_df"] = F.l1_loss(
                self.normalize_df(line_df)
                * df_gt_mask_view0
                * padding_mask_view0,
                self.normalize_df(deeplsd_line_df)
                * df_gt_mask_view0
                * padding_mask_view0,
                # only supervise in line neighborhood
                reduction="none",
            ).mean(dim=(1, 2))


            losses["total"] += (
                self.conf.training.loss.loss_weights.one_view_line_df_weight
                * losses["one_view_line_df"]
            )

        # Use BCE, WeightedBCE or Focal Loss for point position loss
        if self.conf.training.loss.use_one_view_kp_loss:
            if self.conf.training.loss.kp_loss_name == "distill": 
                keypoint_scoremap_loss = self.dad_distil.get_kl_divergence(data, prediction_dict["keypoint_and_junction_score_map"])
            elif self.conf.training.loss.kp_loss_name == "distill_dad":
                ground_heatmap = self.dad_model(data)["heatmap"]
                keypoint_scoremap_loss = self.loss_fn(
                    prediction_dict["keypoint_and_junction_score_map"] * padding_mask_view0,
                    ground_heatmap * padding_mask_view0,
                ).mean(dim=(1, 2))
            elif "distill_dad_superpoint" in self.conf.training.loss.kp_loss_name:
                with torch.no_grad():  
                    ground_heatmap1 = self.dad_model(data)["heatmap"]
                    ground_heatmap2 = self.superpoint_model(data)["heatmap"]
                    ground_heatmap = torch.max(ground_heatmap1, ground_heatmap2)
                keypoint_scoremap_loss = self.loss_fn(
                    prediction_dict["keypoint_and_junction_score_map"] * padding_mask_view0,
                    ground_heatmap * padding_mask_view0,
                ).mean(dim=(1, 2))
            else:
                keypoint_scoremap_loss = self.loss_fn(
                    prediction_dict["keypoint_and_junction_score_map"] * padding_mask_view0,
                    gt_dict_view0["superpoint_heatmap"] * padding_mask_view0,
                ).mean(dim=(1, 2))

            losses["one_view_kp_scoremap"] = keypoint_scoremap_loss
            losses["total"] += (
                self.conf.training.loss.loss_weights.keypoint_weight
                * keypoint_scoremap_loss
            )

        # If training descriptors: decide between one-view and two-view node
        if self.conf.training.train_descriptors.use_one_view_loss:
            # in case of one view: generate gt descriptors to directly supervise using l1 loss
            data = {
                **data,
                **self.get_groundtruth_descriptors(
                    {
                        "keypoints": prediction_dict["keypoints_raw"],
                        "image": gt_dict_view0["image"],
                    }
                ),
            }
            keypoint_descriptor_loss = F.l1_loss(
                prediction_dict["descriptors"],
                data["aliked_descriptors"],
                reduction="none",
            ).mean(dim=(1, 2))
            losses["one_view_kp_descriptors"] = keypoint_descriptor_loss
            losses["total"] += (
                self.conf.training.loss.loss_weights.one_view_descriptor_weight
                * keypoint_descriptor_loss
            )

        # Calculate two view loss (Sparse NRE for descriptors if wanted)
        if (
            self.conf.training.train_descriptors.use_two_view_loss
            and self.conf.training.two_view
        ):
            # Two Way sparse NRE computation

            # best match in kp of A - for each kp projected from B to A
            matches_B_to_A = compute_matches(
                prediction_dict["keypoints"],
                prediction_dict["keypoints1"],
                H,
                best_match_only=True,
            )

            # best match in kp of B - for each kp projected from A to B
            matches_A_to_B = compute_matches(
                prediction_dict["keypoints1"],
                prediction_dict["keypoints"],
                H_inv,
                best_match_only=True,
            )

            # returns overall mean scalar, need to repeat on batch dim for total loss.
            keypoint_descriptor_lossBA = sparse_nre_loss(
                prediction_dict["descriptors"],  # (B, N_A, Dim)
                prediction_dict["descriptors1"],  # (B, N_B, Dim)
                matches_B_to_A,  # (M, 3)
            )

            # returns overall mean scalar, need to repeat on batch dim for total loss.
            keypoint_descriptor_lossAB = sparse_nre_loss(
                prediction_dict["descriptors1"],  # (B, N_B, Dim)
                prediction_dict["descriptors"],  # (B, N_A, Dim)
                matches_A_to_B,  # (M, 3)
            )
            overall_mean = (keypoint_descriptor_lossBA + keypoint_descriptor_lossAB) / 2
            # repeat mean loss across batch dimension to have same mean later on consolidation
            losses["two_view_kp_descriptors"] = overall_mean.repeat(img.shape[0])
            losses["total"] += (
                self.conf.training.loss.loss_weights.two_view_descriptor_weight
                * losses["two_view_kp_descriptors"]
            )

        # Two view df consistency loss
        if self.conf.training.two_view and self.conf.training.loss.use_two_view_df_loss:
            # img1 to img0
            warped_df_1_to_0 = self.warp_data(
                df=prediction_dict["line_distancefield1"],
                angle=data["view1"]["cache"]["deeplsd_angle_field"],
                H=H_inv,
                ps=tuple(prediction_dict["line_distancefield"].shape[1:]),
            ).squeeze(1)
            # valid mask - warp image of ones from view1 to view0. Padding with 0 around warped part gives mask
            valid_mask_1_to_0 = warp_perspective(
                torch.ones_like(
                    prediction_dict["line_distancefield1"].unsqueeze(1),
                    device=prediction_dict["line_distancefield1"].device,
                ),
                H_inv,
                tuple(prediction_dict["line_distancefield1"].shape[1:]),
                mode="nearest",
            ).squeeze(1)

            loss_1to0 = F.l1_loss(  # TODO: use df-gt-mask in two-view consistency loss? (could introduce bias)
                self.normalize_df(prediction_dict["line_distancefield"])
                * df_gt_mask_view0
                * padding_mask_view0
                * valid_mask_1_to_0,
                self.normalize_df(warped_df_1_to_0)
                * df_gt_mask_view0
                * padding_mask_view0
                * valid_mask_1_to_0,
                reduction="none",
            ).mean(
                dim=(1, 2)
            )

            # img0 to img1
            warped_df_0to1 = self.warp_data(
                df=prediction_dict["line_distancefield"],
                angle=gt_dict_view0["deeplsd_angle_field"],  # Note: view0 angle field
                H=H,  # Note: H instead of H_inv
                ps=tuple(prediction_dict["line_distancefield1"].shape[1:]),
            ).squeeze(1)
            # valid mask for 0->1 warping
            valid_mask_0to1 = warp_perspective(
                torch.ones_like(
                    prediction_dict["line_distancefield"].unsqueeze(1),
                    device=prediction_dict["line_distancefield"].device,
                ),
                H,  # Note: H instead of H_inv
                tuple(prediction_dict["line_distancefield"].shape[1:]),
                mode="nearest",
            ).squeeze(1)
            # compute loss
            loss_0to1 = F.l1_loss(
                self.normalize_df(prediction_dict["line_distancefield1"])
                * df_gt_mask_view1
                * padding_mask_view0
                * valid_mask_0to1,
                self.normalize_df(warped_df_0to1)
                * df_gt_mask_view1
                * padding_mask_view0
                * valid_mask_0to1,
                reduction="none",
            ).mean(dim=(1, 2))

            losses["two_view_line_df"] = (loss_0to1 + loss_1to0) / 2
            losses["total"] += (
                self.conf.training.loss.loss_weights.two_view_line_df_weight
                * losses["two_view_line_df"]
            )

        # soft argmax loss (Only applicable with two-view pipeline)
        if (
            self.conf.training.two_view
            and self.conf.training.loss.refinement_radius > 0
            and self.conf.training.loss.loss_weights.softargmax_weight > 0
        ):
            loc_loss = soft_argmax_only_loss(
                pred["keypoint_and_junction_score_map0"],
                pred["keypoint_and_junction_score_map1"],
                gt_dict_view0["keypoints"],
                gt_dict_view0["keypoint_scores"] > 0,
                H,
                self.conf.training.loss.refinement_radius,
            )
            losses["soft_argmax_kp_loss"] = loc_loss
            losses["total"] += (
                self.conf.training.loss.loss_weights.softargmax_weight * loc_loss
            )

        # add metrics if in validation mode
        if not self.training:
            metrics = self.metrics(pred, data)
        return losses, metrics


    def weighted_bce_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Implementation of the weighted BCE loss to cope with class imbalance between keypoint- and non-keypoint pixels.
        We use this loss for leaning the keypoint heatmap.
        """
        epsilon = 1e-6
        return -self.lambda_valid_kp * target * torch.log(prediction + epsilon) - (
            1 - target
        ) * torch.log(1 - prediction + epsilon)

    def focal_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Implementation of the full-focal loss to cope with class imbalance between keypoint- and non-keypoint pixels and
        to focus more on hard examples. We use this loss for leaning the keypoint heatmap.
        """
        alpha = self.conf.training.loss.kp_loss_parameters.focal_alpha
        gamma = self.conf.training.loss.kp_loss_parameters.focal_gamma
        epsilon = 1e-6  # Small value to avoid log(0)

        # Compute the positive and negative parts of the focal loss
        pos_part = (
            -alpha * torch.pow(1 - prediction, gamma) * torch.log(prediction + epsilon)
        )
        neg_part = (
            -(1 - alpha)
            * torch.pow(prediction, gamma)
            * torch.log(1 - prediction + epsilon)
        )

        # Combine the parts to get the total loss
        loss = target * pos_part + (1 - target) * neg_part
        return loss

    def warp_data(self, df, angle, H, ps: list):
        h, w = df.shape[1:3]
        ps = tuple(ps)

        # Warp the closest point on a line
        pix_loc = (
            torch.stack(
                torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij"), dim=-1
            )
            .to(df.device)
            .float()
        )

        warped_dfs = []

        for i in range(df.shape[0]):
            with torch.no_grad():  # TODO: could be batched or made simpler to warp the distance field?
                offset = df[i][:, :, None] * torch.stack(
                    [torch.sin(angle[i]), torch.cos(angle[i])], dim=-1
                )
            closest = pix_loc + offset
            warped_closest = warp_points_torch(
                closest.reshape(-1, 2).unsqueeze(0), H[i], inverse=False
            ).reshape(h, w, 2)
            warped_pix_loc = warp_points_torch(
                pix_loc.reshape(-1, 2).unsqueeze(0), H[i], inverse=False
            ).reshape(h, w, 2)

            offset_norm = torch.linalg.norm(offset, dim=-1)
            zero_offset = offset_norm < 1e-3
            offset_norm[zero_offset] = 1
            scaling = (
                torch.linalg.norm(warped_closest - warped_pix_loc, dim=-1) / offset_norm
            )
            scaling[zero_offset] = 0

            # Warp the DF
            warped_df = warp_perspective(
                df[i][None, None], H[i].unsqueeze(0), ps, mode="bilinear"
            ).squeeze()
            warped_scaling = warp_perspective(
                scaling[None, None], H[i].unsqueeze(0), ps, mode="bilinear"
            ).squeeze()
            warped_df *= warped_scaling

            warped_dfs.append(warped_df)

        return torch.stack(warped_dfs)

    def get_groundtruth_descriptors(self, pred: dict) -> torch.Tensor:
        """
        Takes keypoints from predictions + computes ground-truth descriptors for it.
        """
        assert (
            pred.get("image", None) is not None
            and pred.get("keypoints", None) is not None
        )

        if "aliked" in self.conf.training.train_descriptors.gt_aliked_model:
            with torch.no_grad():
                descriptors = self.aliked_lw(pred)
        else:
            with torch.no_grad():
                descriptors = {}
                descriptors["aliked_descriptors"] = self.descriptor_gt({"image": pred["image"]}, pred["keypoints"])

        return descriptors

    def load_pretrained_aliked_elements(self) -> None:
        """
        Loads ALIKED weights for backbone encoder, score_head(SMH) and SDDH
        """
        # Load state-dict of wanted aliked-model
        aliked_state_url = aliked_checkpoint_url.format(self.conf.aliked_model_name)
        aliked_state_dict = torch.hub.load_state_dict_from_url(
            aliked_state_url, map_location="cpu"
        )
        # change keys
        for k, _ in list(aliked_state_dict.items()):
            if k.startswith("block") or k.startswith("conv"):
                change_dict_key(aliked_state_dict, k, f"encoder_backbone.{k}")
            elif k.startswith("score_head"):
                if not self.conf.training.pretrain_kp_decoder:
                    del aliked_state_dict[k]
                else:
                    change_dict_key(
                        aliked_state_dict, k, f"keypoint_and_junction_branch.{k}"
                    )
            elif k.startswith("desc_head"):
                change_dict_key(aliked_state_dict, k, f"descriptor_branch.{k[10:]}")
            else:
                continue

        # load values
        self.load_state_dict(aliked_state_dict, strict=False)

    def state_dict(self, *args, **kwargs):
        """
        Custom state dict to exclude aliked_lw module from checkpoint.
        """
        sd = super().state_dict(*args, **kwargs)
        # don't store lightweight aliked model for descriptor gt computation
        if self.conf.training.train_descriptors.use_one_view_loss:
            for k in list(sd.keys()):
                if k.startswith("aliked_lw"):
                    del sd[k]
        return sd

    @staticmethod
    def get_pr(
        pred_kp: torch.Tensor, gt_kp: torch.Tensor, tol: int = 3
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the precision and recall, based on GT KP.
        """
        if len(gt_kp) == 0:
            precision = float(len(pred_kp) == 0)
            recall = 1.0
        elif len(pred_kp) == 0:
            precision = 1.0
            recall = float(len(gt_kp) == 0)
        else:
            dist = torch.norm(pred_kp[:, None] - gt_kp[None], dim=2)
            close = (dist < tol).float()
            precision = close.max(dim=1)[0].mean()
            recall = close.max(dim=0)[0].mean()
        return precision, recall

    def metrics(self, pred: dict, data: dict) -> dict:
        """
        Compute evaluation metrics for points. Also for lines if they are contained in the output
        Args:
            pred: dict, containing predictions made by the model
            data: dict containing image data and ground truth

        Returns: dict, containing the computed metrics
        """
        return {}

    # Utility methods for line distance-field for (de)normalization
    def normalize_df(self, df: torch.Tensor) -> torch.Tensor:
        return -torch.log(df / self.conf.line_neighborhood + 1e-6)

    def denormalize_df(self, df_norm: torch.Tensor) -> torch.Tensor:
        return torch.exp(-df_norm) * self.conf.line_neighborhood


    def get_numer_of_parameters(self):
        def count_parameters(model: nn.Module):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        return (count_parameters(self.distance_field_branch) +
                count_parameters(self.descriptor_branch) +
                count_parameters(self.keypoint_and_junction_branch) +
                count_parameters(self.encoder_backbone))