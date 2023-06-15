import cv2
import os
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from collections import deque

class Mask_video_editor():
    def __init__(self, video_path, resize=480, max_memory=100, use_length_ratio=1) -> None:
        self.mask_object_list = []
        self.max_memory = max_memory
        self.video_memory_frames = deque()
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)/(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/resize))
        self.height = resize
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.use_length_ratio = use_length_ratio

        print(f"video:\n\t fps: {self.fps}, width: {self.width}, height: {self.height}, num_frames: {self.num_frames}")
        
    def modify_video_frame_by_frame(self, mask_dir, video_name="test.mp4", objects_edit=None):
        mask_paths = sorted([os.path.join(mask_dir, img_name) for img_name in os.listdir(mask_dir)])        
        output_video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (self.width,self.height))

        for frame_idx, mask_path in tqdm(enumerate(mask_paths[:int(self.use_length_ratio*self.num_frames)])):
            ret, org_frame = self.cap.read()
            mask_img = cv2.imread(mask_path)

            org_frame = cv2.resize(org_frame, (self.width, self.height))
            mask_img = cv2.resize(mask_img, (self.width, self.height)) # , interpolation = cv2.INTER_NEAREST 

            # self.update_video_memory_frame(org_frame)

            for object_idx, object in enumerate(self.mask_object_list):
                object.update_memory_frame(mask_img=mask_img, org_frame=org_frame)
                if object_idx==0:
                    img = object.get_current_frame_object()
                    org_frame = object.make_afterimage_edit(org_frame = org_frame, fps=self.fps)
                    # org_frame = object.make_afterimage_edit(org_frame = img, fps=self.fps)

                    output_video.write(org_frame)
        output_video.release()

    def add_mask_oject(self, mask_dir, detect_new_object_every_n=-1):
        assert self.num_frames == len(os.listdir(mask_dir)), f"number of images:{len(os.listdir(mask_dir))} in mask dir and video frames:{self.num_frames} not correct!"
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
                        color_dict.update({len(self.mask_object_list): frame[yx[0]][yx[1]]})
                        self.mask_object_list.append(Mask_object(self.num_frames, color=frame[yx[0]][yx[1]]))
                        if len(color_dict) == color_num:
                            break

            # print(f"total objects: {len(color_dict)}")
            # detect objects mask

            # for object_idx, color in color_dict.items():
            #     mask = cv2.inRange(frame, color, color)
            #     self.mask_object_list[object_idx].update_mask_frame(frame_idx=i, mask=mask)

    def update_video_memory_frame(self, frame):
        self.video_memory_frames.append(frame)

    def get_current_video_memory_frame(self, idx_from_cuurent=-1):
        return self.video_memory_frames[idx_from_cuurent]

class Mask_object():
    def __init__(self, num_frames, color, max_memory=100, use_object_img=True) -> None:
        self.max_memory = max_memory
        self.color = color
        self.use_object_img = use_object_img
        self.mask_memory_frames = deque(maxlen=max_memory)
        self.object_memory_frames = deque(maxlen=max_memory)

    def update_memory_frame(self, mask_img, org_frame):
        mask = cv2.inRange(mask_img, self.color, self.color)
        self.mask_memory_frames.append(mask)

        if self.use_object_img:
            object_img = cv2.bitwise_and(org_frame, org_frame, mask=mask) 
            self.object_memory_frames.append(object_img)

    def get_current_frame_object_mask(self, idx_from_cuurent=-1):
        return self.mask_memory_frames[idx_from_cuurent]
    
    def make_afterimage_edit(self, org_frame, fps, afterimage_interval_ratio=0.33, residual_time=0.2):
        frame_interval = int(residual_time*fps*afterimage_interval_ratio)
        from_current_list = range(frame_interval, min(len(self.mask_memory_frames), int(residual_time*fps)), frame_interval)[::-1]
        if len(from_current_list)==0:
            return org_frame
        alpha_for_object_list = np.linspace(0.01, 1, len(from_current_list), endpoint=False)
        print(alpha_for_object_list)

        for from_current, alpha_for_object in zip(from_current_list, alpha_for_object_list):
            alpha_for_org = (1 - self.get_current_frame_object_mask(-from_current)*alpha_for_object)
            for i in range(3):
                org_frame[:, :, i] = alpha_for_org*org_frame[:, :, i] + (1-alpha_for_org)*self.get_current_frame_object(-from_current)[:, :, i]

        return org_frame.astype(np.uint8)

    def get_current_frame_object(self, idx_from_cuurent=-1):
        return self.object_memory_frames[idx_from_cuurent]
