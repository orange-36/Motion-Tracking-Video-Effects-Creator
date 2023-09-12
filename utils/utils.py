from matplotlib import pyplot as plt
import numpy as np
import torch

def show_img(img, figsize=(5, 5)):
    plt.figure(figsize=figsize)
    if img.shape[-1]==3:
        plt.imshow(img, interpolation='nearest')
    else: 
        plt.imshow(img, cmap='gray')
    plt.show()

def get_lens_flare(radius=5):
    rows, cols = 1+2*radius, 1+2*radius
    flare_effect = torch.zeros(size=(rows, cols))
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i-radius)**2 + (j-radius)**2)
            if(distance<=radius):
                flare_effect[i][j] = radius-distance
    # all value in range [0, 1]
    flare_effect = flare_effect/radius
    return flare_effect