import cv2
import os
import typing
import math
import numpy as np
import torch
from collections import deque
from typing import List, Deque, Dict, Tuple
from utils.utils import show_img

class BasicObject():
    def __init__(self, color, device, fps, max_memory=100, use_object_img=True) -> None:
        self.max_memory = max_memory
        self.color = color
        self.device = device
        self.fps = fps
        self.use_object_img = use_object_img
        self.use_object_centroids = False
        self.object_centroids: Deque[Tuple[int, int]] = deque(maxlen=max_memory)
        self.mask_memory_frames: Deque[torch.tensor] = deque(maxlen=max_memory)
        self.object_memory_frames: Deque[torch.tensor] = deque(maxlen=max_memory)

    def get_mask_frame(self, mask_img: torch.tensor, org_frame: torch.tensor)-> torch.tensor:
        return None

    def update_memory_frame(self, mask_img: torch.tensor, org_frame: torch.tensor)-> None: 
        mask = self.get_mask_frame(mask_img=mask_img, org_frame=org_frame)
        if mask is None:
            return 
        if self.use_object_centroids:
            _, labeled_image, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
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
        else:
            self.object_centroids.append([])

        mask = torch.tensor(mask.reshape(mask.shape[0], mask.shape[1], 1), dtype=torch.bool).to(self.device)
        object_img = torch.where(mask, org_frame, 0).to(self.device)

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

    def get_mask_memory_frames(self) -> Deque[torch.tensor]:
        return self.mask_memory_frames

    def get_object_memory_frames(self) -> Deque[torch.tensor]:
        return self.object_memory_frames
    
    def get_object_centroids(self) -> Deque[Tuple[int, int]]:
        return self.object_centroids
    
class MaskObject(BasicObject):
    def __init__(self, color, device, fps, max_memory=100, use_object_img=True) -> None:
        super().__init__(color, device, fps, max_memory, use_object_img)

    def get_mask_frame(self, mask_img: torch.tensor, org_frame: torch.tensor)-> torch.tensor:
        mask = cv2.inRange(mask_img, self.color, self.color)
        return mask

    def update_memory_frame(self, mask_img: torch.tensor, org_frame: torch.tensor)-> None: 
        mask = cv2.inRange(mask_img, self.color, self.color)
        if self.use_object_centroids:
            _, labeled_image, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
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
        else:
            self.object_centroids.append([])

        mask = torch.tensor(mask.reshape(mask.shape[0], mask.shape[1], 1), dtype=torch.bool).to(self.device)
        object_img = torch.where(mask, org_frame, 0).to(self.device)

        self.mask_memory_frames.append(mask)
        self.object_memory_frames.append(object_img)
    # def update_memory_frame(self, mask_img, org_frame) -> None:
    #     mask = cv2.inRange(mask_img, self.color, self.color)
        

class DetectedObject(BasicObject):
    def __init__(self, color_lower, color_upper, device, fps, max_memory=100, use_object_img=True) -> None:
        super().__init__(color_lower, device, fps, max_memory, use_object_img)
        self.color_lower = np.array(color_lower, dtype=np.uint8)
        self.color_upper = np.array(color_upper, dtype=np.uint8)

    def get_mask_frame(self, mask_img: torch.tensor, org_frame: torch.tensor)-> torch.tensor:
        mask = cv2.inRange(org_frame.cpu().numpy(), self.color_lower, self.color_upper)
        return mask

    # def update_memory_frame(self, mask_img: torch.tensor, org_frame: torch.tensor) -> None:
    #     mask = cv2.inRange(org_frame.cpu().numpy(), self.color_lower, self.color_upper)
    #     if self.use_object_centroids:
    #         _, labeled_image, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
    #         max_area = -1
    #         max_area_label = -1
    #         for label, stat in enumerate(stats[1:], start=1):
    #             area = stat[4]  # stat[4]: area
    #             if area > max_area:
    #                 max_area = area
    #                 max_area_label = label

    #         # get centroid
    #         if(max_area_label==-1):
    #             self.object_centroids.append([])
    #         else:
    #             centroid_x, centroid_y = centroids[max_area_label]
    #             self.object_centroids.append([(centroid_x, centroid_y)])
    #     else:
    #         self.object_centroids.append([])

    #     mask = torch.tensor(mask.reshape(mask.shape[0], mask.shape[1], 1), dtype=torch.bool).to(self.device)
    #     object_img = torch.where(mask, org_frame, 0).to(self.device)

    #     self.mask_memory_frames.append(mask)
    #     self.object_memory_frames.append(object_img)