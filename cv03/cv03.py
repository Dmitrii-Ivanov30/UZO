import os
import cv2
import numpy as np

os.environ["XDG_SESSION_TYPE"] = "xcb"

angle_degrees = -5
angle = np.radians(angle_degrees)
image_bgr = cv2.imread("../data/cv03_robot.bmp")
y_size, x_size = image_bgr.shape[0], image_bgr.shape[1]

y_size_new = int(abs(x_size * np.sin(angle)) + abs(y_size * np.cos(angle)))
x_size_new = int(abs(x_size * np.cos(angle)) + abs(y_size * np.sin(angle)))

rotated_img = np.zeros((y_size_new, x_size_new, 3), dtype=np.uint8)

center_x, center_y = x_size // 2, y_size // 2
center_x_new, center_y_new = x_size_new // 2, y_size_new // 2
for y in range(y_size_new):
    for x in range(x_size_new):
        y_old = (y - center_y_new) * np.cos(-1 * angle) + (x - center_x_new) * np.sin(-1 * angle) + center_y
        x_old = (x - center_x_new) * np.cos(-1 * angle) - (y - center_y_new) * np.sin(-1 * angle) + center_x

        # Interpolace pomoci metody nejblizsich sousedu
        y_old, x_old = int(round(y_old)), int(round(x_old))

        if 0 <= x_old < x_size and 0 <= y_old < y_size:
            rotated_img[y, x] = image_bgr[y_old, x_old]

cv2.imshow("original image", image_bgr)
cv2.imshow("rotated image", rotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
