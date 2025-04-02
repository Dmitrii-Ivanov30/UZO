import matplotlib.pyplot as plt
import numpy as np

def show_plt():
    manager = plt.get_current_fig_manager()
    manager.resize(1200, 800)
    plt.show()

def plot_images_and_spectrum(images, spectrums, names, cmaps = None):
    size = len(images)
    fig, ax = plt.subplots(size, 2)
    if cmaps is None:
        cmaps = ("gray", ) * size
    for img_index in range(size):
        img_clr_map = ax[img_index, 0].imshow(images[img_index], cmap=cmaps[img_index])
        ax[img_index, 0].set_title(names[img_index])
        if cmaps[img_index] == "jet":
            fig.colorbar(img_clr_map, ax=ax[img_index, 0])
        spectrum_clr_map = ax[img_index, 1].imshow(np.log(np.abs(spectrums[img_index]) + 1), cmap="jet")
        ax[img_index, 1].set_title("Spectrum")
        fig.colorbar(spectrum_clr_map, ax=ax[img_index, 1])

    show_plt()

def hist_segmentation(image: np.ndarray, prah: int=255//2, prah_min: int=0) -> np.ndarray:
    """
    Implementuje algoritmus segmentace pomoci histogramu
    :param image: půvidní obrázek
    :param prah: hodnota prahu
    :param prah_min: hodnota dolniho prahu
    :return:
    """
    hist, bins = np.histogram(image, bins=256, range=(0, 256))
    hist[:prah_min] = 0
    hist[prah_min:prah] = 1
    hist[prah:] = 0
    return hist[image]
