import cv2
import os
import typing
from typing import Deque, List
import math
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from collections import deque
from matplotlib import pyplot as plt
from torchvision.transforms.functional import rotate
from torchvision.transforms import Grayscale
from mask_object import BasicObject, MaskObject, DetectedObject
from effects import *


class Mask_Video_Editor():
    def __init__(self, video_path, mask_dir=None, resize_h=480, max_memory=100, use_length_ratio=1, efficiency_mode=True, realtime_display_enabled = False) -> None:
        self.efficiency_mode = efficiency_mode
        self.max_memory = max_memory
        self.video_memory_frames = deque()
        self.video_path = video_path
        self.mask_dir = mask_dir
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)/(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/resize_h))
        self.height = resize_h
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.use_length_ratio = use_length_ratio
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.object_list = [BasicObject(color=np.array([0, 0, 0]), device=self.device, fps=self.fps)]
        self.object_effects: List[List[BasicEffect]] = [[]]
        self.realtime_display_enabled = realtime_display_enabled
        print(f"video:\n\t fps: {self.fps}, width: {self.width}, height: {self.height}, num_frames: {self.num_frames}, device: {self.device}")
        
    def modify_video_frame_by_frame(self, video_name="test.mp4") -> None:
        if self.mask_dir is not None:
            mask_paths = sorted([os.path.join(self.mask_dir, img_name) for img_name in os.listdir(self.mask_dir)])        
        output_video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (self.width,self.height))
        self.cap = cv2.VideoCapture(self.video_path)

        for frame_idx in tqdm(range(int(self.use_length_ratio*self.num_frames))):
            ret, org_frame = self.cap.read()

            org_frame = torch.tensor(cv2.resize(org_frame, (self.width, self.height)), dtype=torch.uint8).to(self.device)
            background = org_frame.clone()

            if self.mask_dir is not None:
                mask_img = cv2.imread(mask_paths[frame_idx])
                mask_img = cv2.resize(mask_img, (self.width, self.height)) 
            else:
                mask_img = None

            ##### Edit each object sequentially #####
            for object_id, object in enumerate(self.object_list):
                object.update_memory_frame(mask_img=mask_img, org_frame=org_frame)
                
                for effect in self.object_effects[object_id]:
                    effect.object_mask_prepocess(frame_idx=frame_idx)

                for effect in self.object_effects[object_id]:
                    background = effect.perform_editing(org_frame=background, frame_idx=frame_idx)
            background = background.cpu().numpy()
            output_video.write(background)
            
            if self.realtime_display_enabled:
                cv2.imshow('Video', background)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        output_video.release()
        if self.realtime_display_enabled:
            cv2.destroyAllWindows()
        self.clear_effects()

    def add_mask_object(self, mask_dir, detect_new_object_every_n=-1) -> None:
        assert self.num_frames == len(os.listdir(mask_dir)), f"number of images:{len(os.listdir(mask_dir))} in mask dir and video frames:{self.num_frames} not correct!"
        self.mask_dir = mask_dir
        img_paths = sorted([os.path.join(mask_dir, img_name) for img_name in os.listdir(mask_dir)])
        color_dict = {}
        for i, img_path in tqdm(enumerate(img_paths[:int(self.use_length_ratio*self.num_frames)])):
            frame = cv2.imread(img_path)
            frame = cv2.resize(frame, (self.width, self.height), interpolation = cv2.INTER_NEAREST)

            # update objects
            if i==0 or detect_new_object_every_n>0 and i%detect_new_object_every_n==0:
                color_num = Image.open(img_path).getcolors()
                yxs = np.column_stack(np.where(frame>0))

                for yx in yxs:
                    for value in color_dict.values():
                        if (frame[yx[0]][yx[1]] == value).all():
                            break
                    else:
                        color_dict.update({len(self.object_list): frame[yx[0]][yx[1]]})
                        self.object_list.append(MaskObject(color=frame[yx[0]][yx[1]], device=self.device, fps=self.fps))
                        self.object_effects.append([])
                        if len(color_dict) == color_num:
                            break

    def add_detected_object(self, color_upper, color_lower) -> None:
        self.object_list.append(DetectedObject(color_upper=color_upper, color_lower=color_lower, device=self.device, fps=self.fps))
        self.object_effects.append([])

    def get_object(self, index) -> BasicObject: 
        return self.object_list[index]
    
    def show_objects(self) -> BasicObject:
        print("Object 0 is for the background or to process the entire original image")
        for i, object in enumerate(self.object_list):
            print(i)
            object.show_color()
    
    def update_video_memory_frame(self, frame) -> None:
        self.video_memory_frames.append(frame)

    def add_effect(self, object_id: int, effect: BasicEffect) -> None:
        self.object_effects[object_id].append(effect)
        effect.set_attr(fps=self.fps, device=self.device, object=self.object_list[object_id], width=self.width, height=self.height, max_memory=self.max_memory)
        self.object_list[object_id].set_config(effect.config_setting())

    def clear_effects(self) -> None:
        for object in self.object_list:
            object.clear()
        for effects in self.object_effects:
            effects.clear()