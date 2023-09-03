import cv2
import os
import typing
import math
import numpy as np
import torch
from collections import deque
from torchvision.transforms.functional import rotate
from utils.utils import show_img

class Mask_Object():
    def __init__(self, num_frames, color, device, max_memory=100, use_object_img=True) -> None:
        self.max_memory = max_memory
        self.color = color
        self.device = device
        self.use_object_img = use_object_img
        self.mask_memory_frames = deque(maxlen=max_memory)
        self.object_memory_frames = deque(maxlen=max_memory)
        self.edits = {}

    def update_memory_frame(self, mask_img, org_frame):
        mask = cv2.inRange(mask_img, self.color, self.color)
        mask = torch.tensor(mask.reshape(mask.shape[0], mask.shape[1], 1), dtype=torch.bool).to(self.device)

        if self.use_object_img:
            object_img = torch.where(mask, org_frame, 0).to(self.device)
            matadate = self.get_edits().get("kaleidoscope", False)
            if matadate:
                multiple = matadate["multiple"]
                center = matadate["center"]
                angle = 360//multiple
                # show_img(object_img, figsize=(5, 5))
                object_img = object_img.permute(2, 0, 1)
                for degree in range(angle, 361, angle):
                    copy = rotate(object_img, angle=degree, center=center)
                    object_img = torch.bitwise_or(object_img, copy)
                    angle = angle*2
                object_img = object_img.permute(1, 2, 0)
                # show_img(copy.permute(1, 2, 0), figsize=(5, 5))
                # show_img(object_img, figsize=(5, 5))
                mask = object_img > 0
                # print(mask.dtype)
                
        self.mask_memory_frames.append(mask)
        self.object_memory_frames.append(object_img)

    def show_color(self):
        print(self.color)
        color_image = np.array([[self.color for j in range(10)] for i in range(10)])
        show_img(color_image, img_name="color_image.png")

    def update_edits(self, edit):
        self.edits.update(edit)

    def get_edits(self) -> dict:
        return self.edits

    def get_mask_memory_frames(self):
        return self.mask_memory_frames

    def get_object_memory_frames(self):
        return self.object_memory_frames