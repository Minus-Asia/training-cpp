import numpy as np
import cv2

img = cv2.imread('image.jpg')
print(img.shape)
# Crop image
image_arr = img[0:487, 0:365]

cv2.imshow("image_cropped", image_arr)
cv2.imshow("original_image", img)
cv2.waitKey()