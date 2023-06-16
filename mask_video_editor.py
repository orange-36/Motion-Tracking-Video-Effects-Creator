import cv2
import os
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from collections import deque

class Mask_video_editor():
    def __init__(self, video_path, resize=480, max_memory=100, use_length_ratio=1, efficiency_mode=True) -> None:
        self.efficiency_mode = efficiency_mode
        self.mask_object_list = []
        self.max_memory = max_memory
        self.video_memory_frames = deque()
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)/(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/resize))
        self.height = resize
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.use_length_ratio = use_length_ratio
        self.rainbow_colors = np.array([[0, 0, 255], [0, 67, 255], [0, 135, 255], [0, 173, 255], [0, 211, 255], [5, 233, 238], [10, 255, 222], [10, 255, 191], [10, 255, 161], [81, 255, 85], [153, 255, 10], 
                                        [204, 247, 10], [255, 239, 10], [250, 182, 15], [245, 125, 20], [250, 67, 54], [255, 10, 88], [255, 10, 139], [255, 10, 190], [127, 5, 222]], 
                                        dtype=np.uint8)  # BGR Red to Rurple
        self.edit_mapping = {"AfterImage": self.make_afterimage_edit,
                             "Light_track" : self.make_light_track_edit
                             }

        print(f"video:\n\t fps: {self.fps}, width: {self.width}, height: {self.height}, num_frames: {self.num_frames}")
        
    def modify_video_frame_by_frame(self, mask_dir, video_name="test.mp4", objects_edit=None):
        mask_paths = sorted([os.path.join(mask_dir, img_name) for img_name in os.listdir(mask_dir)])        
        output_video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (self.width,self.height))
        self.cap = cv2.VideoCapture(self.video_path)

        for frame_idx, mask_path in enumerate(tqdm((mask_paths[:int(self.use_length_ratio*self.num_frames)]))):
            ret, org_frame = self.cap.read()
            # print(ret)
            mask_img = cv2.imread(mask_path)

            org_frame = cv2.resize(org_frame, (self.width, self.height))
            mask_img = cv2.resize(mask_img, (self.width, self.height)) # , interpolation = cv2.INTER_NEAREST 

            # self.update_video_memory_frame(org_frame)

            for object_idx, object in enumerate(self.mask_object_list):
                object.update_memory_frame(mask_img=mask_img, org_frame=org_frame)
                for edit, para in objects_edit[object_idx].items():
                    # print(f"Object {object_idx}: {objects_edit[object_idx]}")
                    # img = object.get_object_memory_frames()
                    org_frame = self.edit_mapping[edit](object, org_frame = org_frame, fps=self.fps, frame_idx=frame_idx, **para)
                    # org_frame = object.make_afterimage_edit(org_frame = img, fps=self.fps)

            output_video.write(org_frame)
        output_video.release()
        for object in self.mask_object_list:
            object.get_mask_memory_frames().clear()
            object.get_object_memory_frames().clear()

    def add_mask_oject(self, mask_dir, detect_new_object_every_n=-1):
        assert self.num_frames == len(os.listdir(mask_dir)), f"number of images:{len(os.listdir(mask_dir))} in mask dir and video frames:{self.num_frames} not correct!"
        img_paths = sorted([os.path.join(mask_dir, img_name) for img_name in os.listdir(mask_dir)])
        self.mask_object_list = []
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

    def make_afterimage_edit(self, mask_object, org_frame, fps, frame_idx, afterimage_interval_ratio=0.33, residual_time=0.2, alpha_start=0.1, alpha_end=1):
        frame_interval = int(residual_time*fps*afterimage_interval_ratio)
        # if frame_interval < 1:
        #     print(frame_interval)
        from_current_list = range(frame_interval, min(len(mask_object.mask_memory_frames), int(residual_time*fps)), frame_interval)[::-1]
        if len(from_current_list)==0:
            return org_frame
        alpha_for_object_list = np.linspace(alpha_start, alpha_end, len(from_current_list), endpoint=False)


        for from_current, alpha_for_object in zip(from_current_list, alpha_for_object_list):
            if self.efficiency_mode:
                org_frame = cv2.addWeighted(org_frame, 1, mask_object.get_object_memory_frames()[-from_current], alpha_for_object, 0)
            else:
                org_frame = np.where((mask_object.get_mask_memory_frames()[-from_current]), 
                                    alpha_for_object*mask_object.get_object_memory_frames()[-from_current] + (1-alpha_for_object)*org_frame, org_frame)

        return org_frame.astype(np.uint8)
    
    def make_light_track_edit(self, mask_object, org_frame, fps, frame_idx, afterimage_interval_ratio=0.33, residual_time=0.2, alpha_start=0.1, alpha_end=1, gradient=True, rainbow_round_time=-1, color=[255, 255, 255]):
        frame_interval = int(residual_time*fps*afterimage_interval_ratio)
        if frame_interval < 1:
            frame_interval = 1
        from_current_list = range(frame_interval, min(len(mask_object.mask_memory_frames), int(residual_time*fps)), frame_interval)[::-1]
        if len(from_current_list)==0:
            return org_frame
        alpha_for_object_list = np.linspace(alpha_start, alpha_end, len(from_current_list), endpoint=False)

        if rainbow_round_time>0:
            aver_rainbow_color_change_frame = self.fps*rainbow_round_time/len(self.rainbow_colors)
            rainbow_colors_num = len(self.rainbow_colors)

        for from_current, alpha_for_object in zip(from_current_list, alpha_for_object_list):
            if rainbow_round_time>0:
                color_track = self.rainbow_colors[int((frame_idx-from_current)//aver_rainbow_color_change_frame)%rainbow_colors_num]
            else:
                color_track = np.array(color, dtype=np.uint8)

            if gradient:
                org_frame = np.where((mask_object.get_mask_memory_frames()[-from_current]), 
                                        alpha_for_object*color_track + (1-alpha_for_object)*org_frame, org_frame)
            else:
                org_frame = np.where((mask_object.get_mask_memory_frames()[-from_current]), 
                                        color_track, org_frame)


        return org_frame.astype(np.uint8)

class Mask_object():
    def __init__(self, num_frames, color, max_memory=100, use_object_img=True) -> None:
        self.max_memory = max_memory
        self.color = color
        self.use_object_img = use_object_img
        self.mask_memory_frames = deque(maxlen=max_memory)
        self.object_memory_frames = deque(maxlen=max_memory)

    def update_memory_frame(self, mask_img, org_frame):
        mask = cv2.inRange(mask_img, self.color, self.color)
        self.mask_memory_frames.append(mask.reshape(mask.shape[0], mask.shape[1], 1))

        if self.use_object_img:
            object_img = cv2.bitwise_and(org_frame, org_frame, mask=mask) 
            self.object_memory_frames.append(object_img)

    def get_mask_memory_frames(self):
        return self.mask_memory_frames

    def get_object_memory_frames(self):
        return self.object_memory_frames