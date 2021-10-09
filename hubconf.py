import torch
from torchvision.models.resnet import resnet50

from unet_model import UNet

dependencies = ["torch", "torchvision"]


def simple_unet(pretrained=True, **kwargs):
    """
    Originally from ${HOME}/projects/experiments/active/gan-seg/src/outputs/segmentation/bigbigan_dual_run2--12pnucjb/2021-03-08_15-08-32/ganseg/12pnucjb/checkpoints/epoch=4-step=9999.ckpt
    Now renamed to 12pnucjb-step-9999.pth

    Args:
        pretrained (pretrained, optional): Pretrained model. Defaults to True.

    Returns:
        UNet: Segmentation model
    """
    model = UNet(out_channels=2).eval()
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://github.com/greeneggsandyaml/uss/releases/download/v0.0.1/12pnucjb-step-9999.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
    return model


if __name__ == "__main__":
    simple_unet()