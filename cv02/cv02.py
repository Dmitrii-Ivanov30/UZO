import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

plt.ion()
clear = lambda: os.system('cls')
clear()
plt.close('all')

def roi_hist(image_hue):
    hist, _ = np.histogram(image_hue, 180, (0, 180))
    hist = hist.astype("float")
    hist /= hist.max()
    return hist

cap = cv2.VideoCapture('../data/cv02_hrnecek.mp4')
image_to_follow = cv2.imread('../data/cv02_vzor_hrnecek.bmp')
image_to_follow_hue = cv2.cvtColor(image_to_follow, cv2.COLOR_BGR2HSV)[:, :, 0]
hist = roi_hist(image_to_follow_hue)

def find_coordinates(p):
    xs, ys = np.meshgrid(np.arange(p.shape[1]), np.arange(p.shape[0]))
    p_y = p * ys
    p_x = p * xs
    p_sum = np.sum(p)
    return (int(np.round(np.sum(p_y) / p_sum)),
            int(np.round(np.sum(p_x) / p_sum)))

while True:
    ret, bgr = cap.read()
    if not ret:
        break
    hue = cv2.cvtColor(bgr, cv2.COLOR_RGB2HSV)[:, :, 0]
    p = np.zeros_like(hue)
    for y in range(hue.shape[0]):
        for x in range(hue.shape[1]):
            p[y, x] = hist[hue[y, x]]
    # y_center, x_center = find_coordinates(p)
    # y1, x1 = y_center - int(image_to_follow.shape[0]/2), x_center - int(image_to_follow.shape[1]/2)
    # y2, x2 = y_center + int(image_to_follow.shape[0]/2), x_center + int(image_to_follow.shape[1]/2)
    y1, x1 = find_coordinates(p)
    y2, x2 = y1 + image_to_follow.shape[0], x1 + image_to_follow.shape[1]
    cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0))
    cv2.imshow('Image', bgr)
    # hist = roi_hist(hue[y1:y2, x1:x2])
    key = 0xFF & cv2.waitKey(30)
    if key == 27:
        break

cv2.destroyAllWindows()