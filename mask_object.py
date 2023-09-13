import cv2
import os
import typing
import math
import numpy as np
import torch
from collections import deque
from typing import List, Deque
from utils.utils import show_img
from effects import *

class Mask_Object():
    def __init__(self, num_frames, color, device, fps, max_memory=100, use_object_img=True) -> None:
        self.max_memory = max_memory
        self.color = color
        self.device = device
        self.fps = fps
        self.use_object_img = use_object_img
        self.use_object_centroids = False
        self.kaleidoscope = False
        self.object_centroids: Deque[Tuple[int, int]] = deque(maxlen=max_memory)
        self.mask_memory_frames: Deque[torch.tensor] = deque(maxlen=max_memory)
        self.object_memory_frames: Deque[torch.tensor] = deque(maxlen=max_memory)

    def update_memory_frame(self, mask_img, org_frame):
        mask = cv2.inRange(mask_img, self.color, self.color)
        if self.use_object_centroids:
            _, labeled_image, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
            # self.object_centroids.append(centroids)
            # print(centroids)

            # find largest area
            max_area = -1
            max_area_label = -1
            for label, stat in enumerate(stats[1:], start=1):
                area = stat[4]  # stat[4]: area
                if area > max_area:
                    max_area = area
                    max_area_label = label

            # get centroid
            if(max_area_label==-1):
                self.object_centroids.append([])
            else:
                centroid_x, centroid_y = centroids[max_area_label]
                self.object_centroids.append([(centroid_x, centroid_y)])


        mask = torch.tensor(mask.reshape(mask.shape[0], mask.shape[1], 1), dtype=torch.bool).to(self.device)
        object_img = torch.where(mask, org_frame, 0).to(self.device)

        for effect in self.get_effects():
            mask, object_img, self.object_centroids = effect.object_mask_prepocess(mask, object_img, self.object_centroids)
                
        self.mask_memory_frames.append(mask)
        self.object_memory_frames.append(object_img)

    def set_config(self, config: Dict[str, int]):
        for key, value in config.items():
            setattr(self, key, value)
    
    def show_color(self):
        print(self.color)
        color_image = np.array([[self.color for j in range(10)] for i in range(10)])
        show_img(color_image)

    def clear(self):
        self.get_mask_memory_frames().clear()
        self.get_object_memory_frames().clear()
        self.object_centroids.clear()
        self.use_object_centroids = False
        self.kaleidoscope = False

    def get_effects(self) -> List[BasicEffect]:
        return self.effects

    def get_mask_memory_frames(self):
        return self.mask_memory_frames

    def get_object_memory_frames(self):
        return self.object_memory_frames
    
    def get_object_centroids(self):
        return self.object_centroids