from typing import Deque, Dict, Tuple
import torch
import math
import numpy as np
from collections import Counter
from torchvision.transforms.functional import rotate, adjust_brightness, adjust_contrast, affine
from torchvision.transforms import Grayscale
from utils.utils import get_lens_flare, show_img

class BasicEffect:
    def __init__(self) -> None:
        self.fps = None
        self.device = None

    def config_setting(self) -> Dict[str, int]:
        return {}

    def object_mask_prepocess(self, mask: torch.BoolTensor, object_img: torch.tensor, object_centroids: Deque[Tuple[int, int]]) -> None:
        return mask, object_img, object_centroids

    def perform_editing(self, org_frame: torch.tensor, frame_idx: int, mask_memory_frames: Deque[torch.BoolTensor], object_memory_frames: Deque[torch.tensor], **kwargs) -> torch.tensor:
        return org_frame

class AfterimageEffect(BasicEffect):
    def __init__(self, afterimage_interval_ratio=0.33, residual_time=0.2, alpha_start=0.1, alpha_end=1) -> None:
        super().__init__()
        self.afterimage_interval_ratio = afterimage_interval_ratio
        self.residual_time = residual_time
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end

    def perform_editing(self, org_frame: torch.tensor, frame_idx: int, mask_memory_frames: Deque[torch.BoolTensor], object_memory_frames: Deque[torch.tensor], **kwargs):
        frame_interval = math.ceil(self.residual_time*self.fps*self.afterimage_interval_ratio)
        # if frame_interval < 1:
        #     print(frame_interval)
        from_current_list = range(frame_interval, min(len(mask_memory_frames), int(self.residual_time*self.fps)), frame_interval)[::-1]
        if len(from_current_list)==0:
            return org_frame
        alpha_for_object_list = np.linspace(self.alpha_start, self.alpha_end, len(from_current_list), endpoint=False)


        for from_current, alpha_for_object in zip(from_current_list, alpha_for_object_list):
            # print(org_frame)
            draw_track = (alpha_for_object*object_memory_frames[-from_current] + (1-alpha_for_object)*org_frame).type(torch.uint8)
            org_frame = torch.where((object_memory_frames[-from_current]>0), 
                                draw_track, org_frame)
            # print(draw_track)

        return org_frame
    
class LightTrackEffect(BasicEffect):
    def __init__(self, afterimage_interval_ratio=0.33, residual_time=0.2, alpha_start=0.1, alpha_end=1, gradient=True, rainbow_round_time=-1, color=[255, 255, 255]) -> None:
        super().__init__()
        self.afterimage_interval_ratio = afterimage_interval_ratio
        self.residual_time = residual_time
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.gradient = gradient    
        self.rainbow_round_time = rainbow_round_time    
        self.color = color          
        self.rainbow_colors = torch.tensor([[0, 0, 255], [0, 67, 255], [0, 135, 255], [0, 173, 255], [0, 211, 255], [5, 233, 238], [10, 255, 222], [10, 255, 191], [10, 255, 161], [81, 255, 85], [153, 255, 10], 
                                        [204, 247, 10], [255, 239, 10], [250, 182, 15], [245, 125, 20], [250, 67, 54], [255, 10, 88], [255, 10, 139], [255, 10, 190], [127, 5, 222]], 
                                        dtype=torch.uint8) # BGR Red to Rurple
        
    def perform_editing(self, org_frame: torch.tensor, frame_idx: int, mask_memory_frames: Deque[torch.BoolTensor], object_memory_frames: Deque[torch.tensor], **kwargs):
        frame_interval = int(self.residual_time*self.fps*self.afterimage_interval_ratio)
        if frame_interval < 1:
            frame_interval = 1
        from_current_list = range(frame_interval, min(len(mask_memory_frames), int(self.residual_time*self.fps)), frame_interval)[::-1]
        if len(from_current_list)==0:
            return org_frame
        alpha_for_object_list = np.linspace(self.alpha_start, self.alpha_end, len(from_current_list), endpoint=False)

        if self.rainbow_round_time>0:
            aver_rainbow_color_change_frame = self.fps*self.rainbow_round_time/len(self.rainbow_colors)
            rainbow_colors_num = len(self.rainbow_colors)

        for from_current, alpha_for_object in zip(from_current_list, alpha_for_object_list):
            if self.rainbow_round_time>0:
                color_track = self.rainbow_colors[int((frame_idx-from_current)//aver_rainbow_color_change_frame)%rainbow_colors_num].to(self.device)
            else:
                color_track = torch.tensor(self.color, dtype=torch.uint8).to(self.device)

            if self.gradient:

                draw_track = (alpha_for_object*color_track + (1-alpha_for_object)*org_frame).type(torch.uint8)
                org_frame = torch.where((mask_memory_frames[-from_current]), 
                                        draw_track, org_frame)
            else:
                org_frame = torch.where((mask_memory_frames[-from_current]), 
                                        color_track, org_frame)

        return org_frame #.type(torch.uint8)
    

class KaleidoscopeEffect(BasicEffect):
    def __init__(self, multiple: int = 8, center: Tuple[int, int] = (400, 200)) -> None:
        super().__init__()
        # self.multiple = multiple
        self.center = center
        self.angle = 360//multiple

    def object_mask_prepocess(self, mask: torch.BoolTensor, object_img: torch.tensor, object_centroids: Deque[Tuple[int, int]]) -> None:
        # show_img(object_img, figsize=(5, 5))
        object_img = object_img.permute(2, 0, 1)
        angle = self.angle
        for degree in range(angle, 361, angle):
            copy = rotate(object_img, angle=degree, center=self.center)
            object_img = torch.bitwise_or(object_img, copy)
            angle = angle*2
        object_img = object_img.permute(1, 2, 0)
        # show_img(copy.permute(1, 2, 0), figsize=(5, 5))
        # show_img(object_img, figsize=(5, 5))
        mask = object_img > 0
        # print(mask.dtype)
        return mask, object_img, object_centroids
    

class GrayscaleEffect(BasicEffect):
    def __init__(self) -> None:
        super().__init__()

    def perform_editing(self, org_frame: torch.tensor, frame_idx: int, mask_memory_frames: Deque[torch.BoolTensor], object_memory_frames: Deque[torch.tensor], **kwargs):
        org_frame = org_frame.permute(2, 0, 1)
        org_frame = Grayscale(3)(org_frame)
        org_frame = org_frame.permute(1, 2, 0)
        return org_frame

class AdjustBrightnessAndContrast(BasicEffect):
    def __init__(self, brightness_factor=1, contrast_factor=1) -> None:
        super().__init__()
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
    
    def perform_editing(self, org_frame: torch.tensor, frame_idx: int, mask_memory_frames: Deque[torch.BoolTensor], object_memory_frames: Deque[torch.tensor], **kwargs):
        org_frame = org_frame.permute(2, 0, 1)
        if self.brightness_factor!=1:
            org_frame = adjust_brightness(img=org_frame, brightness_factor=self.brightness_factor)
        if self.contrast_factor!=1:
            org_frame = adjust_contrast(img=org_frame, contrast_factor=self.contrast_factor)
        org_frame = org_frame.permute(1, 2, 0)
        return org_frame

class LensFlareEffect(BasicEffect):
    def __init__(self, radius=50, test = False, center_brightness=10, apply_entire_track=True, rendering_effect=False, afterimage_interval_ratio=0.33, residual_time=0.2, alpha_start=0.1, alpha_end=1) -> None:
        super().__init__()
        self.radius = radius
        self.lens_flare = (center_brightness*get_lens_flare(radius=self.radius)).type(dtype=torch.uint8).to(self.device).view(2*radius+1, 2*radius+1, 1)

        self.afterimage_interval_ratio = afterimage_interval_ratio
        self.residual_time = residual_time
        self.apply_entire_track = apply_entire_track        
        self.rendering_effect = rendering_effect
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.test = test

    def config_setting(self) -> Dict[str, int]:
        return {"use_object_centroids": True}

    def perform_editing(self, org_frame: torch.tensor, object_centroids: Deque[Tuple[int, int]], **kwargs):
        frame_interval = math.ceil(self.residual_time*self.fps*self.afterimage_interval_ratio)
        # if frame_interval < 1:
        #     print(frame_interval)
        if self.apply_entire_track:
            from_current_list = range(1, min(len(object_centroids), int(self.residual_time*self.fps)), frame_interval)[::-1]
            if len(from_current_list)==0:
                return org_frame
            alpha_for_object_list = np.linspace(self.alpha_start, self.alpha_end, len(from_current_list), endpoint=True)
        else:
            from_current_list = [1]
            alpha_for_object_list = [1]

        all_lens_flare = torch.zeros((org_frame.shape[0], org_frame.shape[1], 1), dtype=torch.uint8).to(self.device)
        for from_current, alpha_for_object in zip(from_current_list, alpha_for_object_list):
            # print(org_frame)
            centroids_in_one_frame = object_centroids[-from_current]
            # print(f"from_current: {from_current} alpha_for_object {alpha_for_object}")
            for (center_x, center_y) in centroids_in_one_frame:
                one_lens_flare = torch.zeros((org_frame.shape[0], org_frame.shape[1], 1)).to(self.device)
                upper = np.ceil(min(center_y, self.radius))
                lower = np.floor(min(org_frame.shape[0]-center_y, self.radius))
                left  = np.ceil(min(center_x, self.radius))
                right = np.floor(min(org_frame.shape[1]-center_x, self.radius))
                one_lens_flare[int(center_y-upper): int(center_y+lower), int(center_x-left): int(center_x+right)] = self.lens_flare[int(self.radius-upper): int(self.radius+lower), int(self.radius-left): int(self.radius+right)] 
                one_lens_flare = (alpha_for_object*one_lens_flare).type(torch.uint8)
                if self.rendering_effect:
                    all_lens_flare = torch.bitwise_or(input=all_lens_flare, other=one_lens_flare)
                else:
                    all_lens_flare = torch.max(one_lens_flare, all_lens_flare)

                # show_img(all_lens_flare.cpu())
                # print(Counter(all_lens_flare.flatten().tolist()))
        org_frame = org_frame + all_lens_flare.type(torch.float16)
        org_frame = torch.clamp(org_frame, min=0, max=255).type(torch.uint8)
        
        return org_frame