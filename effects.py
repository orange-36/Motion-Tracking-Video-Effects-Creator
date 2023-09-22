from typing import Deque, Dict, Tuple
import torch
import math
import numpy as np
import cv2
from collections import Counter
from collections import deque

import torch.nn.functional as F
from torchvision.transforms.functional import rotate, adjust_brightness, adjust_contrast, affine
from torchvision.transforms import Grayscale
from utils.utils import get_lens_flare, show_img, rotate_point, rainbow_colors
from mask_object import BasicObject
from style_transfer.style_transfer import run_style_transfer
from style_transfer.CCPL import net
import torch.nn as nn



class BasicEffect:
    def __init__(self, max_memory=100) -> None:
        self.fps: float = None
        self.device: torch.device = None
        self.object: BasicObject = None
        self.width: int = None
        self.height: int = None
        self.max_memory = max_memory
        self.effect_memory_frames: Deque = deque(maxlen=max_memory)

    def set_attr(self, fps: float, device: torch.device, object: BasicObject, width: int, height: int, max_memory: int) -> None:
        self.fps = fps
        self.device = device
        self.object = object
        self.width = width
        self.height = height
        self.max_memory = max_memory

    def config_setting(self) -> Dict[str, int]:
        return {}
    
    def object_mask_prepocess(self, frame_idx: int) -> None:
        pass

    def perform_editing(self, org_frame: torch.tensor, frame_idx: int) -> torch.tensor:
        return org_frame

class AfterimageEffect(BasicEffect):
    def __init__(self, afterimage_interval_ratio=0.33, residual_time=0.2, alpha_start=0.1, alpha_end=1) -> None:
        super().__init__()
        self.afterimage_interval_ratio = afterimage_interval_ratio
        self.residual_time = residual_time
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end

    def perform_editing(self, org_frame: torch.tensor, frame_idx: int) -> torch.tensor:
        mask_memory_frames = self.object.get_mask_memory_frames()
        object_memory_frames = self.object.get_object_memory_frames()

        frame_interval = math.ceil(self.residual_time*self.fps*self.afterimage_interval_ratio)
        from_current_list = range(frame_interval, min(len(mask_memory_frames), int(self.residual_time*self.fps)), frame_interval)[::-1]
        if len(from_current_list)==0:
            return org_frame
        alpha_for_object_list = np.linspace(self.alpha_start, self.alpha_end, len(from_current_list), endpoint=False)


        for from_current, alpha_for_object in zip(from_current_list, alpha_for_object_list):
            draw_track = (alpha_for_object*object_memory_frames[-from_current] + (1-alpha_for_object)*org_frame).type(torch.uint8)
            org_frame = torch.where((object_memory_frames[-from_current]>0), 
                                draw_track, org_frame)

        return org_frame
    
class LightTrackEffect(BasicEffect):
    def __init__(self, afterimage_interval_ratio=0.33, residual_time=0.2, alpha_start=0.1, alpha_end=1, gradient=True, rainbow_repeat_interval=-1, color=[255, 255, 255], rainbow_colors=rainbow_colors) -> None:
        super().__init__()
        self.afterimage_interval_ratio = afterimage_interval_ratio
        self.residual_time = residual_time
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.gradient = gradient    
        self.rainbow_repeat_interval = rainbow_repeat_interval    
        self.color = color          
        self.rainbow_colors = torch.tensor(rainbow_colors,  dtype=torch.uint8)
        
    def perform_editing(self, org_frame: torch.tensor, frame_idx: int):
        mask_memory_frames = self.object.get_mask_memory_frames()
        
        frame_interval = int(self.residual_time*self.fps*self.afterimage_interval_ratio)
        if frame_interval < 1:
            frame_interval = 1
        from_current_list = range(frame_interval, min(len(mask_memory_frames), int(self.residual_time*self.fps)), frame_interval)[::-1]
        if len(from_current_list)==0:
            return org_frame
        alpha_for_object_list = np.linspace(self.alpha_start, self.alpha_end, len(from_current_list), endpoint=False)

        if self.rainbow_repeat_interval>0:
            aver_rainbow_color_change_frame = self.fps*self.rainbow_repeat_interval/len(self.rainbow_colors)
            rainbow_colors_num = len(self.rainbow_colors)

        for from_current, alpha_for_object in zip(from_current_list, alpha_for_object_list):
            if self.rainbow_repeat_interval>0:
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

    def object_mask_prepocess(self, frame_idx) -> None:
        mask = self.object.get_mask_memory_frames().pop()
        object_img = self.object.get_object_memory_frames().pop()
        object_centroids = self.object.get_object_centroids()
        # show_img(object_img, figsize=(5, 5))
        object_img = object_img.permute(2, 0, 1)
        angle = self.angle
        if len(object_centroids)>0 and len(object_centroids[-1]):
            centroids_list = object_centroids[-1]
            (x, y) = centroids_list[0]
        for degree in range(angle, 361, angle):
            copy = rotate(object_img, angle=degree, center=self.center)
            object_img = torch.bitwise_or(object_img, copy)
            if len(object_centroids)>0 and len(object_centroids[-1]):
                object_centroids[-1].append(rotate_point(x, y, self.center[0], self.center[1], degree))

            angle = angle*2
        object_img = object_img.permute(1, 2, 0)
        mask = object_img > 0
        self.object.get_mask_memory_frames().append(mask)
        self.object.get_object_memory_frames().append(object_img)
    
class GrayscaleEffect(BasicEffect):
    def __init__(self) -> None:
        super().__init__()

    def perform_editing(self, org_frame: torch.tensor, frame_idx: int):
        org_frame = org_frame.permute(2, 0, 1)
        org_frame = Grayscale(3)(org_frame)
        org_frame = org_frame.permute(1, 2, 0)
        return org_frame

class AdjustBrightnessAndContrast(BasicEffect):
    def __init__(self, brightness_factor=1, contrast_factor=1) -> None:
        super().__init__()
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
    
    def perform_editing(self, org_frame: torch.tensor, frame_idx: int):
        org_frame = org_frame.permute(2, 0, 1)
        if self.brightness_factor!=1:
            org_frame = adjust_brightness(img=org_frame, brightness_factor=self.brightness_factor)
        if self.contrast_factor!=1:
            org_frame = adjust_contrast(img=org_frame, contrast_factor=self.contrast_factor)
        org_frame = org_frame.permute(1, 2, 0)
        return org_frame

class HaloEffect(BasicEffect):
    def __init__(self, radius=50, test = False, center_brightness=10, apply_entire_track=True, rendering_effect=False, afterimage_interval_ratio=0.33, 
                 residual_time=0.2, alpha=0.1, alpha_decay_rated=0.7, bgr_weights=[1, 1, 1], rainbow_repeat_interval=-1, rainbow_colors=rainbow_colors) -> None:
        super().__init__()
        self.radius = radius
        self.lens_flare = (center_brightness*get_lens_flare(radius=self.radius)).to(self.device)
        self.bgr_weights =torch.tensor(bgr_weights).to(self.device)

        self.afterimage_interval_ratio = afterimage_interval_ratio
        self.residual_time = residual_time
        self.apply_entire_track = apply_entire_track        
        self.rendering_effect = rendering_effect
        self.alpha = alpha
        self.alpha_decay_rated = alpha_decay_rated
        self.test = test

        self.rainbow_repeat_interval = rainbow_repeat_interval
        self.rainbow_colors = (torch.tensor(rainbow_colors)/255).to(self.device)
        self.rainbow_colors_num =len(self.rainbow_colors)
        

    def config_setting(self) -> Dict[str, int]:
        return {"use_object_centroids": True}

    def object_mask_prepocess(self, frame_idx) -> None:
        centroids_in_one_frame = self.object.get_object_centroids()[-1]
        all_halo_in_one_frame = torch.zeros((self.height, self.width, 3), dtype=torch.uint8).to(self.device)
        for (center_x, center_y) in centroids_in_one_frame:
            one_lens_flare = torch.zeros((self.height, self.width, 3)).to(self.device)
            center_x, center_y = int(center_x), int(center_y)
            upper = int(np.floor(max(center_y-self.radius, 0)))
            lower = int(np.ceil(min(center_y+self.radius, self.height)))
            left  = int(np.floor(max(center_x-self.radius, 0)))
            right = int(np.ceil(min(center_x+self.radius, self.width)))
            if(lower<0 or upper>self.height or right<0 or left>self.width):
                continue
            pattern_upper = self.radius + upper - center_y
            pattern_lower = self.radius + lower - center_y
            pattern_left = self.radius + left - center_x
            pattern_right = self.radius + right - center_x

            if self.rainbow_repeat_interval>0:
                aver_rainbow_color_change_frame = self.fps*self.rainbow_repeat_interval/self.rainbow_colors_num
                bgr_weights = self.rainbow_colors[int((frame_idx)//aver_rainbow_color_change_frame)%self.rainbow_colors_num].to(self.device)
            else:
                bgr_weights = self.bgr_weights.to(self.device)
            color_lens_flare = bgr_weights* self.lens_flare.to(self.device)
            one_lens_flare[upper: lower, left: right] = color_lens_flare[pattern_upper: pattern_lower, pattern_left: pattern_right] 
            if self.rendering_effect:
                all_halo_in_one_frame = torch.bitwise_or(input=all_halo_in_one_frame, other=one_lens_flare.type(torch.uint8))
            else:
                all_halo_in_one_frame = torch.max(one_lens_flare, all_halo_in_one_frame)
        self.effect_memory_frames.append(all_halo_in_one_frame)

    def perform_editing(self, org_frame: torch.tensor, frame_idx: int):
        object_centroids = self.object.get_object_centroids()
        frame_interval = math.ceil(self.residual_time*self.fps*self.afterimage_interval_ratio)

        if self.apply_entire_track:
            from_current_list = range(1, min(len(object_centroids), int(self.residual_time*self.fps)), frame_interval)[::-1]
            if len(from_current_list)==0:
                return org_frame
            alpha_for_object_list = [1]
            for i in range(1, len(from_current_list)):
                alpha_for_object_list.append(self.alpha_decay_rated* alpha_for_object_list[-1])
            alpha_for_object_list = alpha_for_object_list[::-1]
        else:
            from_current_list = [1]
            alpha_for_object_list = [1]

        all_lens_flare = torch.zeros((org_frame.shape[0], org_frame.shape[1], 3), dtype=torch.uint8).to(self.device)
        for from_current, alpha_for_object in zip(from_current_list, alpha_for_object_list):
            # print(org_frame)
            all_halo_in_one_frame = self.effect_memory_frames[-from_current]
            # print(f"from_current: {from_current} alpha_for_object {alpha_for_object}")
            
            one_lens_flare = (alpha_for_object*all_halo_in_one_frame).type(torch.uint8)
            if self.rendering_effect:
                all_lens_flare = torch.bitwise_or(input=all_lens_flare, other=one_lens_flare)
            else:
                all_lens_flare = torch.max(one_lens_flare, all_lens_flare)

        org_frame = org_frame + all_lens_flare.type(torch.float16)
        org_frame = torch.clamp(org_frame, min=0, max=255).type(torch.uint8)
        
        return org_frame
    
class DilationEffect(BasicEffect):
    def __init__(self, dilation_degree=1) -> None:
        super().__init__()
        # self.multiple = multiple
        kernel_size = 2 * dilation_degree + 1
        self.kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    def object_mask_prepocess(self, frame_idx) -> None:
        mask = self.object.get_mask_memory_frames().pop().cpu().numpy()
        dalited_img = cv2.dilate(mask.astype(np.uint8), self.kernel)
        dalited_img = torch.tensor(dalited_img, dtype=torch.bool).view(self.height, self.width, 1).to(self.device)
        self.object.get_mask_memory_frames().append(dalited_img)


class StyleTransferEffect(BasicEffect):
    def __init__(self, reference_path, afterimage_interval_ratio=0.33, residual_time=0.2, transform_size=512) -> None:
        super().__init__()
        self.afterimage_interval_ratio = afterimage_interval_ratio
        self.residual_time = residual_time
        self.reference_img = cv2.imread(reference_path)
        self.decoder = net.decoder
        self.vgg = net.vgg
        self.network = net.Net(self.vgg, self.decoder, 'art')
        self.SCT = self.network.SCT

        self.SCT.eval()
        self.decoder.eval()
        self.vgg.eval()

        self.decoder.load_state_dict(torch.load('./style_transfer/CCPL/artistic/decoder_iter_160000.pth.tar'))
        self.vgg.load_state_dict(torch.load('./style_transfer/CCPL/models/vgg_normalised.pth'))
        self.SCT.load_state_dict(torch.load('./style_transfer/CCPL/artistic/sct_iter_160000.pth.tar'))
        self.vgg = nn.Sequential(*list(self.vgg.children())[:31]) 

        self.transform_size = transform_size

    def perform_editing(self, org_frame: torch.tensor, frame_idx: int) -> torch.tensor:
        mask_memory_frames = self.object.get_mask_memory_frames()

        frame_interval = math.ceil(self.residual_time*self.fps*self.afterimage_interval_ratio)
        from_current_list = range(1, min(len(mask_memory_frames), int(self.residual_time*self.fps)), frame_interval)[::-1]
        if len(from_current_list)==0:
            return org_frame

        mask = torch.zeros((self.height, self.width, 1), dtype=torch.bool).to(self.device)
        for from_current in from_current_list:
            mask = torch.bitwise_or(mask, mask_memory_frames[-from_current])
            
        org_frame = run_style_transfer(img=org_frame, mask=mask, reference_image=self.reference_img, device=self.device, decoder=self.decoder, SCT=self.SCT, vgg=self.vgg)

        return org_frame