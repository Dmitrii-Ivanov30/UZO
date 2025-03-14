import cv2
from matplotlib import pyplot as plt
import numpy as np

fig, ax = plt.subplots(9, 9, figsize=(20, 20))
for img in range(1, 10):
    distances = {}
    images = {}
    image_bgr = cv2.imread(f"../data/im0{img}.jpg")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    image_hist = cv2.calcHist(image_gray, [0], None, [256], [0, 256])
    for i in range(1, 10):
        path = f'../data/im0{i}.jpg'
        bgr = cv2.imread(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist(gray, [0], None, [256], [0, 256])
        distance = np.linalg.norm(hist - image_hist)
        if distance == 0:
            continue
        images[path] = rgb
        distances[path] = distance
    distances = sorted(distances.items(), key=lambda x: x[1])
    ax[img-1, 0].imshow(image_rgb)
    ax[img-1, 0].axis('off')
    for idx, item in enumerate(distances):
        path = item[0]
        ax[img-1, idx+1].imshow(images[path])
        ax[img-1, idx+1].axis('off')
plt.show()
