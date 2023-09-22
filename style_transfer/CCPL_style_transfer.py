import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
#from .net import decoder,vgg,Net
# import style_transfer.CCPL as CCPL
import sys
sys.path.append(str(Path('./style_transfer/CCPL')))
from .CCPL.function import nor_mean_std, nor_mean


def input_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, SCT, content, style, device, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    vgg = vgg.to(device)
    SCT = SCT.to(device)
    decoder = decoder.to(device)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = SCT(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = SCT(content_f, style_f)
    return decoder(feat)

def run_style_transfer(img, reference_img, decoder, SCT, vgg, transform_size=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content_tf = input_transform(transform_size, False)
    style_tf = input_transform(transform_size, False)

    ## 使用reference_img
    content = content_tf(img)
    style = style_tf(reference_img)
    style = style.to(device).unsqueeze(0)
    content = content.to(device).unsqueeze(0)
    with torch.no_grad():
        output = style_transfer(vgg, decoder, SCT, content, style,
                                device, 1.0)
    return output[0] # dim 0 is batchsize
