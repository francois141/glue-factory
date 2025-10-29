import torch

from gluefactory.models.scalelsd_backbone import build_backbone


class VITBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        print("Creating the VIT backbone")
        self.vit_encoder = build_backbone(
            gray_scale=False,
            use_layer_scale=False,
            enable_attention_hooks=False,
            head_size=[[128]],
            upsample=True,
        )

    def forward(self, inputs):
        output, _ = self.vit_encoder(inputs)
        return output
