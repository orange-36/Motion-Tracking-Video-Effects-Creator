from matplotlib import pyplot as plt
import numpy as np
import torch



 # BGR Red to Rurple
rainbow_colors = [[0, 0, 255], [0, 67, 255], [0, 135, 255], [0, 173, 255], [0, 211, 255], [5, 233, 238], [10, 255, 222], [10, 255, 191], [10, 255, 161], [81, 255, 85], [153, 255, 10], 
                                        [204, 247, 10], [255, 239, 10], [250, 182, 15], [245, 125, 20], [250, 67, 54], [255, 10, 88], [255, 10, 139], [255, 10, 190], [127, 5, 222]]

def show_img(img, figsize=(5, 5)):
    plt.figure(figsize=figsize)
    if img.shape[-1]==3:
        plt.imshow(img, interpolation='nearest')
    else: 
        plt.imshow(img, cmap='gray')
    plt.show()

def get_lens_flare(radius=5, exp_decay=1.5, use_decay_rate=1):
    rows, cols = 1+2*radius, 1+2*radius
    flare_effect = torch.zeros(size=(rows, cols))
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i-radius)**2 + (j-radius)**2)
            if(distance<=radius):
                flare_effect[i][j] = radius-distance
    # all value in range [0, 1]
    flare_effect = flare_effect.view(rows, cols, 1)
    if use_decay_rate==1:
        flare_effect = (flare_effect/radius)**exp_decay
    else:
        flare_effect = torch.pow(use_decay_rate, 10*(radius - flare_effect)/radius)
    return flare_effect

def rotate_point(x, y, cx, cy, angle_degrees):
    angle_radians = np.radians(angle_degrees)

    rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians)],
                                 [np.sin(angle_radians), np.cos(angle_radians)]])

    point_vector = np.array([x - cx, y - cy])

    rotated_vector = np.dot(rotation_matrix, point_vector)
    x_new = rotated_vector[0] + cx
    y_new = rotated_vector[1] + cy

    return x_new, y_new