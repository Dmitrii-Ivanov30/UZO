import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
from scipy.fft import dctn, idctn

from uzo_func.func import show_plt

matplotlib.use('TkAgg')
cmap = plt.get_cmap("jet")
cmap.set_bad("white")  # Set masked values (zeros) to white

gray = cv2.imread("../data/cv04c_robotC.bmp", cv2.IMREAD_GRAYSCALE).astype(np.float32)

#  ULOHA 1

fig, ax = plt.subplots(1, 2)
fft2 = np.fft.fft2(gray)
fft2_log = np.log(np.abs(fft2)+1)
img1 = ax[0].imshow(fft2_log, cmap="jet")
ax[0].set_title("Spectrum fft2")
fig.colorbar(img1, ax=ax[0])

def center_freq(fft):
    height, width = fft.shape
    height_half, width_half = height // 2, width // 2
    fft_centered = np.zeros_like(fft)

    fft_centered[:height_half, :width_half] = fft[height_half:, width_half:]
    fft_centered[:height_half, width_half:] = fft[height_half:, :width_half]
    fft_centered[height_half:, :width_half] = fft[:height_half, width_half:]
    fft_centered[height_half:, width_half:] = fft[:height_half, :width_half]
    return fft_centered

fft_centered = center_freq(fft2)
img2 = ax[1].imshow(np.log(np.abs(fft_centered)+1), cmap="jet")
ax[1].set_title("Spectrum, centered low freq")
fig.colorbar(img2, ax=ax[1])
show_plt()

#  ULOHA 2

dp_1 = cv2.imread("../data/cv04c_filtDP.bmp", cv2.IMREAD_GRAYSCALE) // 255
dp_2 = cv2.imread("../data/cv04c_filtDP1.bmp", cv2.IMREAD_GRAYSCALE) // 255
hp_1 = cv2.imread("../data/cv04c_filtHP.bmp", cv2.IMREAD_GRAYSCALE) // 255
hp_2 = cv2.imread("../data/cv04c_filtHP1.bmp", cv2.IMREAD_GRAYSCALE) // 255

def filter_image(fft, pass_filter):
    fft_pass = fft * pass_filter
    img = np.abs(np.fft.ifft2(fft_pass))
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)  # normalizace
    fft_pass_img = np.log(np.abs(fft_pass) + 1)
    fft_pass_img[fft_pass_img == 0] = np.nan
    return fft_pass, img, fft_pass_img

fft_dp_1, img_dp_1, fft_dp_1_img = filter_image(fft_centered, dp_1)
fft_dp_2, img_dp_2, fft_dp_2_img = filter_image(fft_centered, dp_2)

_, ax = plt.subplots(2, 2)
ax[0, 0].imshow(fft_dp_1_img, cmap=cmap)
ax[0, 0].set_title("Spectrum, Filtr DP")
ax[0, 1].imshow(img_dp_1, cmap="gray")
ax[0, 1].set_title("Result")
ax[1, 0].imshow(fft_dp_2_img, cmap=cmap)
ax[1, 1].imshow(img_dp_2, cmap="gray")
show_plt()

fft_hp_1, img_hp_1, fft_hp_1_img = filter_image(fft_centered, hp_1)
fft_hp_2, img_hp_2, fft_hp_2_img = filter_image(fft_centered, hp_2)

_, ax = plt.subplots(2, 2)
ax[0, 0].imshow(fft_hp_1_img, cmap=cmap)
ax[0, 0].set_title("Spectrum, Filtr HP")
ax[0, 1].imshow(img_hp_1, cmap="gray")
ax[0, 1].set_title("Result")
ax[1, 0].imshow(fft_hp_2_img, cmap=cmap)
ax[1, 1].imshow(img_hp_2, cmap="gray")
show_plt()

#  ULOHA 3

dctS = dctn(gray)
fig, ax = plt.subplots(1, 2)
ax[0].imshow(gray, cmap="gray")
clrs = ax[1].imshow(np.log(np.abs(dctS)), cmap="jet")
fig.colorbar(clrs, ax=ax[1])
ax[1].set_title("DCT Spectrum")
show_plt()

#  ULOHA 4

def dct_limited_calc_and_plot(dct, n):
    # calculate dct spectrum limited to nxn area
    dct_limited = np.zeros_like(dct)
    dct_limited[:n, :n] = dct[:n, :n]
    img = idctn(dct_limited)
    # calculate log and mask zeros for better plot visibility
    dct_limited_img = np.log(np.abs(dct_limited) + 1)
    dct_limited_img[dct_limited_img == 0] = np.nan
    # plot result dct and image
    fig, ax = plt.subplots(1, 2)
    clrs = ax[0].imshow(dct_limited_img, cmap=cmap)
    fig.colorbar(clrs, ax=ax[0])
    ax[0].set_title(f"DCT Spectrum {n}x{n}")
    ax[1].imshow(img, cmap="gray")
    show_plt()

dct_limited_calc_and_plot(dctS, 10)
dct_limited_calc_and_plot(dctS, 30)
dct_limited_calc_and_plot(dctS, 50)

#  ULOHA 5

fig, ax = plt.subplots(9, 9, figsize=(20, 20))
R = 5
for img in range(1, 10):
    distances = {}
    images = {}
    image_bgr = cv2.imread(f"../data/uzo_cv01_IT_im0{img}.jpg")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    image_dct = dctn(image_gray)
    image_dct_vec = image_dct[:R, :R].flatten()
    for i in range(1, 10):
        path = f'../data/uzo_cv01_IT_im0{i}.jpg'
        bgr = cv2.imread(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        dct = dctn(gray)
        dct_vec = dct[:R, :R].flatten()
        distance = np.linalg.norm(dct_vec - image_dct_vec)
        if distance == 0:
            continue
        images[path] = rgb
        distances[path] = distance
    distances = sorted(distances.items(), key=lambda x: x[1])
    ax[img - 1, 0].imshow(image_rgb)
    ax[img - 1, 0].axis('off')
    for idx, item in enumerate(distances):
        path = item[0]
        ax[img - 1, idx + 1].imshow(images[path])
        ax[img - 1, idx + 1].axis('off')
plt.show()
