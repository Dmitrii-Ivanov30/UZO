from typing import Tuple

import cv2
import numpy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from uzo_func.func import hist_segmentation

matplotlib.use('TkAgg')

def barveni_oblasti(image: numpy.ndarray) -> numpy.ndarray:
    counter = 2
    image_padded = np.zeros((image.shape[0] + 1, image.shape[1] + 2), dtype=image.dtype)
    image_padded[1:, 1:-1] = image
    master_values = []  # hlavní hodnoty, které budou v konečném výsleku
    slave_values = [[]]  # druhotné hodnoty, které na konci se budou rovnat hlavním
    # první cyklus
    for y in range(1, image_padded.shape[0]):
        for x in range(1, image_padded.shape[1]-1):
            if image_padded[y, x] == 0:
                continue
            area = image_padded[[y-1, y-1, y-1, y], [x-1, x, x+1, x-1]]
            area_values = np.unique(area)
            if 0 in area_values:
                if area_values.size == 1:
                    image_padded[y, x] = counter
                    counter += 1
                    continue
                else:
                    area_values = area_values[1:]
            image_padded[y, x] = area_values[0]
            if area_values.size > 1:
                master_value = area_values[0]
                index = -1
                if master_value in master_values:
                    index = master_values.index(master_value)
                else:
                    for i in range(len(slave_values)):
                        if master_value in slave_values[i]:
                            index = i
                            break
                if index == -1:
                    index = len(master_values)
                    master_values.append(master_value)
                    slave_values.append([])
                for slave_value in area_values[1:]:
                    if slave_value not in slave_values[index]:
                        slave_values[index].append(slave_value)
    image_unpadded = image_padded[1:, 1:-1]
    # druhý cyklus
    for y in range(image_unpadded.shape[0]):
        for x in range(image_unpadded.shape[1]):
            value = image_unpadded[y, x]
            if value == 0:
                continue
            if value in master_values:
                continue
            for i in range(len(slave_values)):
                if value in slave_values[i]:
                    image_unpadded[y, x] = master_values[i]
                    break
    return (image_unpadded / image_unpadded.max() * 255).astype(int)

def object_coordinates_and_size(image: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    Funkce hledá těžiště a velikost (v pixelech) objektů na obrázku
    :param image: obrázek po průchodu algoritmu barvení obrázku
    :return: tuple s vektory x, y těžišť, a velikosti objektů
    """
    values = np.unique(image)[1:]
    x_coordinates = np.zeros(values.size, dtype=np.int32)
    y_coordinates = np.zeros(values.size, dtype=np.int32)
    size = np.zeros(values.size, dtype=np.int32)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            value = image[y, x]
            if value == 0:
                continue
            x_coordinates[values == value] += x
            y_coordinates[values == value] += y
            size[values == value] += 1
    x_coordinates = np.floor(x_coordinates.astype(np.float32) / size).astype(np.int32)
    y_coordinates = np.floor(y_coordinates.astype(np.float32) / size).astype(np.int32)
    return x_coordinates, y_coordinates, size


if __name__ == '__main__':
    bgr = cv2.imread("../data/cv07_segmentace.bmp").astype(np.float32)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    g = ((rgb[:, :, 1] * 255) / np.sum(rgb, axis=2)).astype(int)
    segmentation = hist_segmentation(g, 105)
    plt.imshow(segmentation)
    plt.show()
    result = barveni_oblasti(segmentation)
    x_coords, y_coords, size = object_coordinates_and_size(result)
    print("(x, y): hodnota mince")
    for i in range(size.size):
        if size[i] >= 4000:
            coin_type = 5
        else:
            coin_type = 1
        print(f"({x_coords[i]}, {y_coords[i]}): {coin_type}")
    plt.imshow(result)
    plt.show()
