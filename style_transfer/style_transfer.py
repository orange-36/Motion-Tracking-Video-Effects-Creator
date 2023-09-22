import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import resize
from . import CCPL_style_transfer

def mask2bbox(mask):
    coords = torch.where(mask != 0)
    if(len(coords[0])==0):
        return (-1, -1, -1, -1)
    r1 = torch.min(coords[0])
    r2 = torch.max(coords[0])
    c1 = torch.min(coords[1])
    c2 = torch.max(coords[1])
    return c1.item(), r1.item(), c2.item(), r2.item()


def run_style_transfer(img, mask, reference_image, device, decoder, SCT, vgg, transform_size=512):
    box_coord = mask2bbox(mask)
    if(box_coord[0]==-1):
        return img
    if mask.ndim == 2:
        mask = mask.type(torch.uint8)*255
    
    crop_img = img[box_coord[1]:box_coord[3], box_coord[0]:box_coord[2]]
    crop_mask = mask[box_coord[1]:box_coord[3], box_coord[0]:box_coord[2]]
    output = CCPL_style_transfer.run_style_transfer(
        Image.fromarray(crop_img.cpu().numpy()),
        reference_img=Image.fromarray(reference_image),
        decoder=decoder,
        SCT=SCT,
        vgg=vgg,
        transform_size = transform_size
    )
    output = resize(output, (box_coord[3]-box_coord[1], box_coord[2]-box_coord[0]), antialias=True)
    output = output.permute(1, 2, 0)

    output = output*255
    output = torch.clamp(input=output, min=0, max=255)
    output = torch.where(crop_mask, output, crop_img)
    img[box_coord[1]:box_coord[3], box_coord[0]:box_coord[2]] = output
    return img
