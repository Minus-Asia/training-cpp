import numpy as np
import cv2


image = cv2.imread("1.bmp")
image = cv2.resize(image, (0,0), fx=0.3, fy=0.3)

#get the image dimensions and calculate the center of image
(h, w) = image.shape[:2]

(cX, cY) = (w // 2, h // 2)

#call function to get Rotation Matrix with 26 degree angel and scale 1.0
M1 = cv2.getRotationMatrix2D((cX, cY), 26, 1.0)
print(M1);
rotated1 = cv2.warpAffine(image, M1, (w, h))
cv2.imshow("Rotated by 26 Degrees with warpAffine", rotated1)


#create M matrix by Numpy
theta = np.radians(360-26)
c, s = np.cos(theta), np.sin(theta)
M2 = np.array(((c, -s, (cX*(1-c) + cY*s)), (s, c, (cY*(1-c) - cX*s))))
print(M2)

# get the coordinates in the form of (0,0),(0,1)...
# the shape is (2, rows*cols)
#day la toa doa cua cac diem ban dau cua anh
orig_coord = np.indices((w, h)).reshape(2,-1)
# stack the rows of 1 to form [x,y,1]
#add them mot chieu vao de de dang nhan voi M matrix
orig_coord_f = np.vstack((orig_coord, np.ones(h*w)))

# apply the transformation by multiplying the transformation matrix with coordinates.
# nhan toa do ban dau voi ma tran chuyen doi de ra duoc ma tran toa do sau khi xoay
transform_coord = np.dot(M2, orig_coord_f)

transform_coord = transform_coord.astype(np.int)
# Keep only the coordinates that fall within the image boundary.
# giu lai cac toa do nam trong w, h cua anh, nhung cai nao nam ngoai thi loai bo
indices = np.all((transform_coord[1]<h, transform_coord[0]<w, transform_coord[1]>=0, transform_coord[0]>=0), axis=0)

#tao ra 1 ma tran rong de chuan bi copy cac diem anh tu toa do ban dau sang toa do xoay
rotated2 = np.zeros_like(image)
# copy cac diem anh tu anh goc toa do ban dau sang toa do moi
rotated2[transform_coord[1][indices], transform_coord[0][indices]] = image[orig_coord[1][indices], orig_coord[0][indices]]

cv2.imshow("rotated2", rotated2)
cv2.waitKey(0)