import numpy
import cv2

img = cv2.imread('image.jpg')
# change channels last to channels first format
data = numpy.moveaxis(img, 2, 0)
print(data.shape)

arr = numpy.array(img, dtype=float)
# divide RGB to 2,4,8
arr[:,:,0] /= 2
arr[:,:,1] /= 4
arr[:,:,2] /= 8

cv2.imshow("divided_image", arr)
cv2.imshow("original_image", img)

cv2.waitKey()
