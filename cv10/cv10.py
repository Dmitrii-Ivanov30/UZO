import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from uzo_func.func import show_plt

if __name__ == '__main__':
    matplotlib.use('TkAgg')
    # Learning phase
    pca_layer_matrix = np.zeros((64*64, 9))
    for i1 in range(1, 4):
        for i2 in range(1, 4):
            gray = cv2.imread(f"../data/p{i1}{i2}.bmp", cv2.IMREAD_GRAYSCALE)
            pca_layer_matrix[:, (i1-1) * 3 + (i2-1)] = gray.flatten().astype(np.float32)
    pca_mean_vector = np.mean(pca_layer_matrix, axis=1, keepdims=True).astype(np.float32)
    pca_layer_matrix -= pca_mean_vector
    cov_matrix = pca_layer_matrix.T @ pca_layer_matrix
    d, e = np.linalg.eig(cov_matrix)
    sorted_indices = np.argsort(d)[::-1]
    e_sorted = e[sorted_indices, :]
    eigen_space = pca_layer_matrix @ e_sorted
    P = eigen_space.T @ pca_layer_matrix

    # Testing phase
    bgr = cv2.imread('../data/unknown.bmp')
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    img = gray.flatten().astype(np.float32)
    img -= pca_mean_vector.flatten()
    P_img = eigen_space.T @ img
    diff = np.linalg.norm(P-P_img, axis=1)
    ind_closest = np.argmin(diff)
    ind_closest = sorted_indices[ind_closest]
    i1 = ind_closest // 3
    i2 = ind_closest % 3
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(rgb)
    ax[0].set_title('Unknown image')
    ax[0].axis('off')
    bgr_closest = cv2.imread(f"../data/p{i1+1}{i2+1}.bmp")
    rgb_closest = cv2.cvtColor(bgr_closest, cv2.COLOR_BGR2RGB)
    ax[1].imshow(rgb_closest)
    ax[1].set_title('Similar image')
    ax[1].axis('off')

    show_plt()
