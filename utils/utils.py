from matplotlib import pyplot as plt

def show_img(img, figsize=(1, 1)):
    plt.figure(figsize=figsize)
    plt.imshow(img, interpolation='nearest')
    plt.show()