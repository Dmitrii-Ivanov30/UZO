from typing import Tuple

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from uzo_func.func import hist_segmentation, show_plt
from cv07.cv07 import barveni_oblasti, object_coordinates_and_size

matplotlib.use('TkAgg')

def eroze(img: np.ndarray, kernel: np.ndarray=np.array([[1, 1]]), anchor: Tuple[int, int]=(0,0)) -> np.ndarray:
    bottom_pixels = kernel.shape[0] - anchor[0]
    top_pixels = anchor[0]
    right_pixels = kernel.shape[1] - anchor[1]
    left_pixels = anchor[1]
    print(bottom_pixels, top_pixels, right_pixels, left_pixels)
    img_eroded = np.zeros_like(img, np.uint8)
    for y in range(top_pixels, img.shape[0]-bottom_pixels+1):
        for x in range(left_pixels, img.shape[1]-right_pixels+1):
            mask = img[y-top_pixels:y+bottom_pixels, x-left_pixels:x+right_pixels]
            if False not in (mask * kernel == kernel):
                img_eroded[y, x] = 1
    return img_eroded

def dilatace(img: np.ndarray, kernel: np.ndarray=np.array([[1, 1]]), anchor: Tuple[int, int]=(0,0)) -> np.ndarray:
    kernel_ones = (kernel==1)
    bottom_pixels = kernel.shape[0] - anchor[0]
    top_pixels = anchor[0]
    right_pixels = kernel.shape[1] - anchor[1]
    left_pixels = anchor[1]
    img_dilated = img.copy()
    for y in range(top_pixels, img.shape[0] - bottom_pixels + 1):
        for x in range(left_pixels, img.shape[1] - right_pixels + 1):
            if img[y, x] == 1:
                img_dilated[y - top_pixels:y + bottom_pixels, x - left_pixels:x + right_pixels][kernel_ones] = 1
    return img_dilated

def otevreni(img: np.ndarray, kernel: np.ndarray=np.array([[1, 1]]), anchor: Tuple[int, int]=(0,0)) -> np.ndarray:
    return dilatace(eroze(img, kernel, anchor), kernel, anchor)

def uzavreni(img: np.ndarray, kernel: np.ndarray=np.array([[1, 1]]), anchor: Tuple[int, int]=(0,0)) -> np.ndarray:
    return eroze(dilatace(img, kernel, anchor), kernel, anchor)



if __name__ == '__main__':
    prahy = [(180, 100), (255, 190)]
    bgr = cv2.imread("../data/cv08_im1.bmp")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(rgb)
    axs[0, 0].set_title("Original")
    axs[0, 0].axis('off')
    segmentation = hist_segmentation(gray, 100)
    axs[0, 1].imshow(segmentation, cmap='gray')
    axs[0, 1].set_title("Segmentace")
    axs[0, 1].axis('off')
    kernel = np.ones((5, 5), np.uint8)
    kernel[[0, 0, 4, 4], [4, 0, 4, 0]] = 0
    otevreni_result = otevreni(segmentation, kernel, (2, 2))
    axs[1, 0].imshow(otevreni_result, cmap='gray')
    axs[1, 0].set_title("Otevreni")
    axs[1, 0].axis('off')
    result_barveni = barveni_oblasti(otevreni_result)
    x_coords, y_coords, size = object_coordinates_and_size(result_barveni)
    axs[1, 1].imshow(rgb)
    axs[1, 1].scatter(x_coords, y_coords, c="green", s=20, marker="x")
    axs[1, 1].set_title("Vysledek")
    axs[1, 1].axis('off')
    show_plt()

    bgr = cv2.imread("../data/cv08_im2.bmp")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    hue_float = np.array(hue, dtype=np.float32)
    hue_norm = np.empty_like(hue_float)
    cv2.normalize(hue_float, dst=hue_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    hue_255 = hue_norm.astype(np.uint8)
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(rgb)
    axs[0, 0].set_title("Original")
    axs[0, 0].axis('off')
    axs[0, 1].imshow(hue_255)
    axs[0, 1].set_title("Hue slozka")
    axs[0, 1].axis('off')
    segmentation = hist_segmentation(hue_255, 255, 190)
    axs[1, 0].imshow(segmentation, cmap='gray')
    axs[1, 0].set_title("Segmentace")
    axs[1, 0].axis('off')
    result_barveni = barveni_oblasti(segmentation)
    x_coords, y_coords, size = object_coordinates_and_size(result_barveni)
    axs[1, 1].imshow(rgb)
    axs[1, 1].scatter(x_coords, y_coords, c="green", s=20, marker="x")
    axs[1, 1].set_title("Vysledek")
    axs[1, 1].axis('off')
    show_plt()
