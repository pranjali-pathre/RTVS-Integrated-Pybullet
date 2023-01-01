import cv2
import numpy as np

# Load image, grayscale, Gaussian blur, adaptive threshold
image = cv2.imread("imgs/00040.png")
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
mask = np.zeros(image.shape, dtype=np.uint8)
gray = 255 - cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# segment red colour from image
gray = cv2.inRange(hsv_image, (0, 100, 100), (10, 255, 255))
# blur = cv2.GaussianBlur(gray, (3,3), 0)
# thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 3)
# Morph open
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
blur = thresh = opening = mask = gray

# Find distorted rectangle contour and draw onto a mask
# cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# rect = cv2.minAreaRect(cnts[0])
# box = cv2.boxPoints(rect)
# box = np.int0(box)
# cv2.drawContours(image,[box],0,(36,255,12),2)
# cv2.fillPoly(mask, [box], (255,255,255))

# # Find corners on the mask
# mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
# corners = cv2.goodFeaturesToTrack(mask, maxCorners=4, qualityLevel=0.5, minDistance=150)

# find corners of white rectangle in binary image of mask

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

for corner in corners:
    x, y = corner.ravel()
    x, y = int(x), int(y)
    x, y = y, x
    cv2.circle(image, (x, y), 8, (255, 120, 255), -1)
    print("({}, {})".format(x, y))

cv2.imshow("thresh", thresh)
cv2.imshow("opening", opening)
cv2.imshow("mask", mask)
cv2.imshow("masked_img", mask.reshape(*mask.shape, 1) * image)
cv2.waitKey(0)
cv2.imshow("image", image)
cv2.waitKey(0)
