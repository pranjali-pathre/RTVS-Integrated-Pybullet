import cv2
import numpy as np


def detect_corners(rgb_img):
    hsv_image = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
    mask = np.zeros(rgb_img.shape, dtype=np.uint8)
    # segment red colour from image
    mask = cv2.inRange(hsv_image, (0, 100, 100), (10, 255, 255))
    whites = np.argwhere(mask == 255)
    top_left = whites.min(axis=0)
    bottom_right = whites.max(axis=0)
    corners = np.array(
        [
            top_left,
            [top_left[0], bottom_right[1]],
            bottom_right,
            [bottom_right[0], top_left[1]],
        ]
    )

    # in x,y format
    corners[:, 0], corners[:, 1] = corners[:, 1], corners[:, 0]
    return corners
