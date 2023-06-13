import cv2
import os
import numpy as np
from PIL import Image
from tqdm.auto import tqdm


class Mask_video_editor():
    def __init__(self, video_path) -> None:
        self.mask_object_list = []
        self.video = cv2.VideoCapture(video_path)
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.num_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

    def update_mask_image(self, mask_dir, detect_new_object_every_n=-1):
        assert self.num_frames == len(os.listdir(mask_dir)), "number of images in mask dir not correct!"
        img_paths = sorted([os.path.join(mask_dir, img_name) for img_name in os.listdir(mask_dir)])
          
        color_dict = {}
        for i, img_path in tqdm(enumerate(img_paths)):
            frame = cv2.imread(img_path)
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
                        self.mask_object_list.append(Mask_object(self.num_frames))
                        if len(color_dict) == color_num:
                            break

            # detect objects mask
            for object_idx, color in color_dict.items():
                mask = cv2.inRange(frame, color, color)
                self.mask_object_list[object_idx].update_mask_frame(frame_num=i, mask=mask)
            
class Mask_object():
    def __init__(self, num_frames) -> None:
        self.mask_frames = [[] for i in range(num_frames)]

    def update_mask_frame(self, frame_num, mask):
        self.mask_frames[frame_num] = mask

    def get_mask_frame(self, frame_num):
        return self.mask_frames[frame_num]
