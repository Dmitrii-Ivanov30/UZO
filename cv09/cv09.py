import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from uzo_func.func import hist_segmentation, show_plt
from cv07.cv07 import barveni_oblasti, object_coordinates_and_size

def tophat(image, kernel):
    opened_image = cv2.dilate(cv2.erode(image, kernel), kernel)
    tophat_image = cv2.subtract(image, opened_image)
    return tophat_image

if __name__ == '__main__':
    matplotlib.use('TkAgg')
    bgr = cv2.imread('../data/cv09_rice.bmp')
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((15, 15), np.uint8)
    segmentation = hist_segmentation(gray, 255, 133)
    hist_og, bins_og = np.histogram(gray, bins=256, range=(0, 256))
    fig, ax = plt.subplots(2,2)
    ax[0, 0].bar(bins_og[:-1], hist_og)
    ax[0, 0].set_title('Histogram original')
    ax[1, 0].imshow(segmentation, "gray")
    ax[1, 0].set_title("segmentation original image")
    ax[1, 0].axis('off')

    tophat_image = tophat(gray, kernel)
    tophat_segmentation = hist_segmentation(tophat_image, 255, 52)
    hist_tophat, bins_tophat = np.histogram(tophat_image, bins=256, range=(0, 256))
    ax[0, 1].bar(bins_tophat[:-1], hist_tophat)
    ax[0, 1].set_title('Histogram tophat')
    ax[1, 1].imshow(tophat_segmentation, "gray")
    ax[1, 1].set_title("segmentation tophat image")
    ax[1, 1].axis('off')
    show_plt()

    colored_image = barveni_oblasti(tophat_segmentation)
    xs, ys, sizes = object_coordinates_and_size(colored_image)

    rice_indices = []
    for i in range(len(sizes)):
        if 30 < sizes[i]:
            rice_indices.append(i)

    plt.imshow(rgb)
    plt.title("Number of rice grains: {}".format(len(rice_indices)))
    for i in rice_indices:
        plt.plot(xs[i], ys[i], 'rx', markersize=7)
    plt.axis('off')
    show_plt()
