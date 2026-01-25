# Integration of PlNet joint points and lines detector: https://github.com/sair-lab/PLNet


import sys
from pathlib import Path

import torch
from ...settings import DATA_PATH
from ..base_model import BaseModel

# Add PLNet to sys.path
plnet_path = Path(__file__).parent.parent.parent.parent / "other/PLNet"
sys.path.append(str(plnet_path))

try:
    from hawp.fsl.config import cfg as model_config
    from hawp.fsl.model.build import build_model
    print("Successfully imported PLNet components from ", plnet_path)
except ImportError as e:
    print(f"Warning: Could not import PLNet from {plnet_path}. Make sure the submodule is initialized and dependencies (yacs, easydict) are installed.")
    print(f"Error: {e}")


class PLNet(BaseModel):
    default_conf = {
        "checkpoint_url": "https://entuedu-my.sharepoint.com/:u:/g/personal/kuan_xu_staff_main_ntu_edu_sg/EbQy7pSPVNFDrP81aloP-O8BA3W0HlOqFsTi6p20KGH9xA?e=mFgVdU&download=1",
        "config_path": "other/PLNet/configs/plnet.yaml",
        "junction_threshold": 0.008,
        "line_threshold": 0.05,
        "max_num_junctions": 512,
        "max_num_lines": 512,
    }

    def _init(self, conf):
        """
        Initialize PlNet model - download model weights if not accessible yet and load model.
        """
        print("Initialize PLNet....")
        root = Path(__file__).parent.parent.parent.parent
        ckpt_path = DATA_PATH / "weights/plnet.pth"
        if not ckpt_path.is_file():
            self.download_model(ckpt_path)

        # Load config
        full_config_path = root / conf.config_path
        model_config.merge_from_file(str(full_config_path))

        # Initialize model
        self.net = build_model(model_config).eval()
        
        # Load weights
        state_dict = torch.load(ckpt_path, map_location="cpu")
        self.net.load_state_dict(state_dict)

    def download_model(self, path):
        import subprocess

        if not path.parent.is_dir():
            path.parent.mkdir(parents=True, exist_ok=True)
        link = self.conf.checkpoint_url
        cmd = ["wget", link, "-O", str(path)]
        print(f"Downloading PLNet model from {link}...")
        subprocess.run(cmd, check=True)

    def _forward(self, data):
        """
        Forward pass of PlNet. Wraps PlNet forward and processes lines and points that are output to glue-factory
        compatible format.
        """
        image = data["image"]
        # PLNet expects grayscale image in [0, 1] range, and normalized with mean/std
        # Glue-factory usually provides image in [0, 1] range
        if image.shape[1] == 3:
            # Convert to grayscale
            scale = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            image = (image * scale).sum(1, keepdim=True)
        
        # PLNet normalization
        # From configs/plnet.yaml:
        # PIXEL_MEAN: [109.73, 103.832, 98.681]
        # PIXEL_STD: [22.275, 22.124, 23.229]
        # However, for grayscale it's often different.
        # Let's check how they do it in predict.py
        # image_ = torch.from_numpy(image_).float()/255.0
        # image_ = image_[None,None].to(args.device)
        # It seems they just divide by 255 if grayscale.
        
        batch_size, _, H, W = image.shape
        
        # Meta information required by PLNet
        metas = []
        for i in range(batch_size):
            metas.append({
                'width': W,
                'height': H,
                'filename': ''
            })

        with torch.no_grad():
            # forward_test returns (outputs, extra_info)
            # PLNet's forward_test only supports batch_size=1 due to annotations[0] usage
            all_juncs = []
            all_junc_scores = []
            all_lines = []
            all_line_scores = []
            all_valid_lines = []

            for i in range(batch_size):
                out, _ = self.net.forward_test(image[i:i+1], [metas[i]])
                
                juncs = out['juncs_pred'] # [N, 2]
                junc_scores = out['juncs_score'] # [N]
                
                # Filter junctions by threshold
                mask = junc_scores > self.conf.junction_threshold
                juncs = juncs[mask]
                junc_scores = junc_scores[mask]
                
                # Keep top-k junctions
                if len(juncs) > self.conf.max_num_junctions:
                    junc_scores, indices = torch.topk(junc_scores, self.conf.max_num_junctions)
                    juncs = juncs[indices]
                
                all_juncs.append(juncs)
                all_junc_scores.append(junc_scores)
                
                # Lines
                lines = out['lines_pred'].reshape(-1, 2, 2) # [M, 2, 2]
                line_scores = out['lines_score'] # [M]
                
                # Filter lines by threshold
                mask = line_scores > self.conf.line_threshold
                lines = lines[mask]
                line_scores = line_scores[mask]
                
                # Keep top-k lines
                if len(lines) > self.conf.max_num_lines:
                    line_scores, indices = torch.topk(line_scores, self.conf.max_num_lines)
                    lines = lines[indices]
                
                all_lines.append(lines)
                all_line_scores.append(line_scores)
                all_valid_lines.append(torch.ones(len(lines), dtype=torch.bool, device=lines.device))

        # We need to pad junctions and lines if we want to batch them, 
        # but glue-factory often handles variable number of points/lines if not batched.
        # For now, let's assume batch_size=1 or use padding if needed.
        # BaseModel usually expects batched tensors.
        
        if batch_size == 1:
            return {
                "keypoints": all_juncs[0][None],
                "keypoint_scores": all_junc_scores[0][None],
                "lines": all_lines[0][None],
                "line_scores": all_line_scores[0][None],
                "valid_lines": all_valid_lines[0][None],
            }
        else:
            # TODO: Implement padding for batch > 1 if necessary
            # For now return list-based if that's acceptable, but usually it's not.
            # Let's stick to batch size 1 for simplicity or use torch.nn.utils.rnn.pad_sequence
            
            # Pad keypoints
            keypoints = torch.nn.utils.rnn.pad_sequence(all_juncs, batch_first=True)
            keypoint_scores = torch.nn.utils.rnn.pad_sequence(all_junc_scores, batch_first=True)
            
            # Pad lines
            lines = torch.nn.utils.rnn.pad_sequence(all_lines, batch_first=True)
            line_scores = torch.nn.utils.rnn.pad_sequence(all_line_scores, batch_first=True)
            valid_lines = torch.nn.utils.rnn.pad_sequence(all_valid_lines, batch_first=True)
            
            return {
                "keypoints": keypoints,
                "keypoint_scores": keypoint_scores,
                "lines": lines,
                "line_scores": line_scores,
                "valid_lines": valid_lines,
            }

    def loss(self, pred, data):
        """
        Loss not needed for pure inference integration.
        """
        raise NotImplementedError