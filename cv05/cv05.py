import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2

from uzo_func.func import show_plt

matplotlib.use('TkAgg')

def metoda_prosteho_prumerovani(img, mask=np.ones((3, 3), dtype=np.float32)/9):
    h, w = img.shape
    filtered_image = img.copy()
    filtered_image[1:h-1, 1:w-1] = np.zeros((h-2, w-2)).astype(np.float32)
    for y in range(1, h-1):
        for x in range(1, w-1):
            filtered_image[y, x] = np.sum(img[y-1:y+2, x-1:x+2] * mask)
    return filtered_image.astype(np.uint8)

def metoda_rotujuci_masky(img, mask=np.ones((3, 3), dtype=np.float32)/9):
    h, w = img.shape
    filtered_image = img.copy()
    filtered_image[2:h - 2, 2:w - 2] = np.zeros((h - 4, w - 4)).astype(np.float32)
    variances = np.zeros_like(img, dtype=np.float32)
    for y in range(1, h-1):
        for x in range(1, w-1):
            variances[y, x] = np.var(img[y-1:y+2, x-1:x+2] * mask)
    for y in range(2, h-2):
        for x in range(2, w-2):
            min_var = np.inf
            for y_step in range(-1, 2):
                for x_step in range(-1, 2):
                    if y_step == 0 and x_step == 0:
                        continue
                    new_x = x + x_step
                    new_y = y + y_step
                    current_var = variances[new_y, new_x]
                    if current_var < min_var:
                        min_var = current_var
                        filtered_image[y, x] = np.sum(img[new_y-1:new_y+2, new_x-1:new_x+2] * mask)
    return filtered_image.astype(np.uint8)

def metoda_medianu(img):
    h, w = img.shape
    filtered_image = img.copy()
    filtered_image[2:h - 2, 2:w - 2] = np.zeros((h - 4, w - 4)).astype(np.float32)
    for y in range(2, h-2):
        for x in range(2, w-2):
            vector_for_median = np.zeros((9,), dtype=np.float32)
            vector_for_median[:5] = img[y, x-2:x+3].flatten()
            vector_for_median[5:7] = img[y-2:y, x].flatten()
            vector_for_median[7:] = img[y+1:y+3, x].flatten()
            filtered_image[y, x] = np.median(vector_for_median)
    return filtered_image.astype(np.uint8)


def plot_images_and_spectrum(img1, img2, spectrum1, spectrum2, method="Filtered"):
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(img1, cmap="gray")
    ax[0, 0].set_title("Original")
    ax[0, 1].imshow(np.log(np.abs(spectrum1) + 1), cmap="jet")
    ax[0, 1].set_title("Spectrum")
    ax[1, 0].imshow(img2, cmap="gray")
    ax[1, 0].set_title(method)
    ax[1, 1].imshow(np.log(np.abs(spectrum2) + 1), cmap="jet")
    ax[1, 1].set_title("Spectrum")

    show_plt()

gray = cv2.imread("../data/cv05_robotS.bmp", cv2.IMREAD_GRAYSCALE).astype(np.float32)

filtered_image = metoda_prosteho_prumerovani(gray)
fft2_og = np.fft.fft2(gray)
fft2_og = np.fft.fftshift(fft2_og)
fft2_filtered = np.fft.fft2(filtered_image)
fft2_filtered = np.fft.fftshift(fft2_filtered)
plot_images_and_spectrum(gray, filtered_image, fft2_og, fft2_filtered, "Average")

filtered_image = metoda_rotujuci_masky(gray)
fft2_filtered = np.fft.fft2(filtered_image)
fft2_filtered = np.fft.fftshift(fft2_filtered)
plot_images_and_spectrum(gray, filtered_image, fft2_og, fft2_filtered, "Rotation mask")

filtered_image = metoda_medianu(gray)
fft2_filtered = np.fft.fft2(filtered_image)
fft2_filtered = np.fft.fftshift(fft2_filtered)
plot_images_and_spectrum(gray, filtered_image, fft2_og, fft2_filtered, "Median")

gray = cv2.imread("../data/cv05_PSS.bmp", cv2.IMREAD_GRAYSCALE).astype(np.float32)

filtered_image = metoda_prosteho_prumerovani(gray)
fft2_og = np.fft.fft2(gray)
fft2_og = np.fft.fftshift(fft2_og)
fft2_filtered = np.fft.fft2(filtered_image)
fft2_filtered = np.fft.fftshift(fft2_filtered)
plot_images_and_spectrum(gray, filtered_image, fft2_og, fft2_filtered, "Average")

filtered_image = metoda_rotujuci_masky(gray)
fft2_filtered = np.fft.fft2(filtered_image)
fft2_filtered = np.fft.fftshift(fft2_filtered)
plot_images_and_spectrum(gray, filtered_image, fft2_og, fft2_filtered, "Rotation mask")

filtered_image = metoda_medianu(gray)
fft2_filtered = np.fft.fft2(filtered_image)
fft2_filtered = np.fft.fftshift(fft2_filtered)
plot_images_and_spectrum(gray, filtered_image, fft2_og, fft2_filtered, "Median")
