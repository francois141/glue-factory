import torch
from faster_pytlsd import params_lsd
from wireframe_distillation.wireframe_net import WireframeNet

from ...settings import DATA_PATH
from ..base_model import BaseModel


def download_model(path):
    import subprocess

    if not path.parent.is_dir():
        path.parent.mkdir(parents=True, exist_ok=True)
    link = "https://raw.githubusercontent.com/iferfra/wireframe-detector/main/checkpoints/checkpoint.pth"
    cmd = ["wget", link, "-O", path]
    print("Downloading Wireframe model...")
    subprocess.run(cmd, check=True)


class WireframeSuarez(BaseModel):
    default_conf = {}

    required_data_keys = ["image"]

    def _init(self, conf):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_name = "wireframe-suarez"

        ckpt = DATA_PATH / "weights" / self.model_name
        if not ckpt.is_file():
            download_model(ckpt)

        config = {"weights": ckpt, "size": [800, 800]}

        self.model = WireframeNet(config)
        self.model.eval().to(device)

    def _forward(self, data):
        results = self.model.forward(data)

        return {
            "lines": torch.tensor(results["line_segments"]).unsqueeze(0),
            "keypoints": results["points"],
            "keypoint_scores": results["scores"],
            "descriptors": results["descriptors"],
        }

    def loss(self, pred, data):
        raise NotImplementedError

    def is_initialized(self):
        return True
