from typing import Deque, Dict, Tuple
import torch
import math
import numpy as np
from torchvision.transforms.functional import rotate

class BasicEffect:
    def __init__(self) -> None:
        pass

    def config_setting(self) -> Dict[str, int]:
        return {}

    def object_mask_prepocess(self, mask: torch.BoolTensor, object_img: torch.tensor) -> None:
        return mask, object_img

    def perform_editing(self, org_frame: torch.tensor, frame_idx: int, mask_memory_frames: Deque[torch.BoolTensor], object_memory_frames: Deque[torch.tensor], fps: int, device: torch.device) -> torch.tensor:
        return org_frame

class AfterimageEffect(BasicEffect):
    def __init__(self, afterimage_interval_ratio=0.33, residual_time=0.2, alpha_start=0.1, alpha_end=1) -> None:
        super().__init__()
        self.afterimage_interval_ratio = afterimage_interval_ratio
        self.residual_time = residual_time
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end

    def perform_editing(self, org_frame, frame_idx, mask_memory_frames: Deque[torch.BoolTensor], object_memory_frames: Deque[torch.tensor], fps: int, device: torch.device):
        frame_interval = math.ceil(self.residual_time*fps*self.afterimage_interval_ratio)
        # if frame_interval < 1:
        #     print(frame_interval)
        from_current_list = range(frame_interval, min(len(mask_memory_frames), int(self.residual_time*fps)), frame_interval)[::-1]
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
        
    def perform_editing(self, org_frame, frame_idx, mask_memory_frames: Deque[torch.BoolTensor], object_memory_frames: Deque[torch.tensor], fps: int, device: torch.device):
        frame_interval = int(self.residual_time*fps*self.afterimage_interval_ratio)
        if frame_interval < 1:
            frame_interval = 1
        from_current_list = range(frame_interval, min(len(mask_memory_frames), int(self.residual_time*fps)), frame_interval)[::-1]
        if len(from_current_list)==0:
            return org_frame
        alpha_for_object_list = np.linspace(self.alpha_start, self.alpha_end, len(from_current_list), endpoint=False)

        if self.rainbow_round_time>0:
            aver_rainbow_color_change_frame = fps*self.rainbow_round_time/len(self.rainbow_colors)
            rainbow_colors_num = len(self.rainbow_colors)

        for from_current, alpha_for_object in zip(from_current_list, alpha_for_object_list):
            if self.rainbow_round_time>0:
                color_track = self.rainbow_colors[int((frame_idx-from_current)//aver_rainbow_color_change_frame)%rainbow_colors_num].to(device)
            else:
                color_track = torch.tensor(self.color, dtype=torch.uint8).to(device)

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

    def object_mask_prepocess(self, mask: torch.BoolTensor, object_img: torch.tensor) -> None:
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
        return mask, object_img