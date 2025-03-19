import cv2
import numpy as np
import matplotlib

from uzo_func.func import plot_images_and_spectrum

matplotlib.use('TkAgg')

def detektor_hran(img, detector_type):

    mask_types = {
        "laplace": [
            np.array([[1, 1, 1],
                      [1, -8, 1],
                      [1, 1, 1]])
        ],
        "sobel": [
            np.array([[1, 2, 1],
                      [0, 0, 0],
                      [-1, -2, -1]]),
            np.array([[0, 1, 2],
                      [-1, 0, 1],
                      [-2, -1, 0]])
        ],
        "kirsch": [
            np.array([[3, 3, 3],
                      [3, 0, 3],
                      [-5, -5, -5]]),
            np.array([[3, 3, 3],
                      [-5, 0, 3],
                      [-5, -5, 3]])
        ]
    }

    masks = mask_types.get(detector_type.lower())
    if masks is None:
        raise TypeError(f"{detector_type} detector type is not supported")
    h, w = img.shape
    img_edges = np.zeros((h-2, w-2), dtype=np.float32)
    values = np.zeros((8,))
    if detector_type.lower() == "laplace":
        mask = masks[0]
        for y in range(h - 2):
            for x in range(w - 2):
                img_edges[y, x] = np.sum(img[y:y+3, x:x+3] * mask)
    else:
        for y in range(h - 2):
            for x in range(w - 2):
                value_id = 0
                for mask in masks:
                    for _ in range(4):
                        # values[value_id] = np.sum(img[y:y+3, x:x+3] * mask) ** 2
                        values[value_id] = np.sum(img[y:y+3, x:x+3] * mask)
                        value_id += 1
                        mask = np.rot90(mask)
                # img_edges[y, x] = np.sqrt(np.sum(values))
                img_edges[y, x] = np.max(values)
    return img_edges

gray = cv2.imread("../data/cv04c_robotC.bmp", cv2.IMREAD_GRAYSCALE).astype(np.float32)

detector_types = ("Laplace", "Sobel", "Kirsch")

for detector in detector_types:
    filtered_image = detektor_hran(gray, detector)
    fft2_og = np.fft.fft2(gray)
    fft2_og = np.fft.fftshift(fft2_og)
    fft2_filtered = np.fft.fft2(filtered_image)
    fft2_filtered = np.fft.fftshift(fft2_filtered)
    plot_images_and_spectrum((gray, filtered_image), (fft2_og, fft2_filtered), ("Original", detector), ("gray", "jet"))
