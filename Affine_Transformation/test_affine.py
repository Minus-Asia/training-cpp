import numpy as np
import cv2


image = cv2.imread("1.bmp")
image = cv2.resize(image, (0,0), fx=0.3, fy=0.3)

#get the image dimensions and calculate the center of image
(h, w) = image.shape[:2]
# h(row) = 614, w (column) = 922

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

#day la toa do cua cac diem ban dau cua anh
# input: width va height cua anh
# ouput: orig_coord co dang
# [[0,0,0,...., 921],[0,1,2,3,...,613]]
# indices(w, h): muc dich de define ra matrix w x h chua thong tin toa do cac diem anh
# thong tin toa do vi du nhu la pixel 1: (0,0) pixel 2: (0,1)... pixel 922 (1,0), pixel 566108: (613, 921)
# reshape (2, -1): reshape de tao ra ma tran co 2 hang (tong so phan tu 1 hang se la 566108 (=614*922))
# shape cua orig_coord = (2, 566108)
# cu the hon ve indices: https://stackoverflow.com/questions/32271331/can-anybody-explain-me-the-numpy-indices
orig_coord = np.indices((w, h)).reshape(2, -1)

# stack the rows of 1 to form [x,y,1]
# add them mot chieu vao de de dang nhan voi M matrix vi ma tran M co dang 2*3
# ma tran luc nay co shape = (3, 566108) vi add them mot hang toan so 1
# input: matrix orig_coord
# output: matrix co shape (3, 566108)
orig_coord_f = np.vstack((orig_coord, np.ones(h*w)))

# apply the transformation by multiplying the transformation matrix with coordinates.
# nhan toa do ban dau voi ma tran chuyen doi de ra duoc ma tran toa do sau khi xoay

# input la ma tran orig_coord_f (3,566108) va ma tran M (2*3) da duoc tinh o tren
# output: transform_coord: la ma tran chua thong tin cac toa do diem se duoc thay doi:
# vi du nhu pixel (0,0) o toa do ban dau sau khi nhan voi M se ra toa do moi (x,y) nao do cua pixel nay
#
transform_coord = np.dot(M2, orig_coord_f)
transform_coord = transform_coord.astype(np.int)

# Keep only the coordinates that fall within the image boundary.
# giu lai cac toa do nam trong w, h cua anh, nhung cai nao nam ngoai thi loai bo
# input: la transform coordinate vua tinh o ben tren. transform_coord nay chua toan bo cac toa do co the nam ngoai (w*h)
# cua buc anh muon xoay, nen can phai so sanh va loai bo no di.
# ouput: la boolean true false cac toa do nam trong vung cua buc anh (w*h)
indices = np.all((transform_coord[1]<h, transform_coord[0]<w, transform_coord[1]>=0, transform_coord[0]>=0), axis=0)

#tao ra 1 ma tran rong de chuan bi copy cac diem anh tu toa do ban dau sang toa do xoay
rotated2 = np.zeros_like(image)
print(orig_coord.shape)
print(transform_coord.shape)
print(indices.shape)

# copy cac diem anh tu anh goc toa do ban dau sang toa do moi
# rotated2[transform_coord[1][indices], transform_coord[0][indices]] = image[orig_coord[1][indices], orig_coord[0][indices]]

# mot cach viet khac cua dong 72 cho de hieu cach copy
for i in range(len(indices)):
    # neu indices bang true thi copy gia tri
    if indices[i] == True:
        # transform_coord[1][i],transform_coord[0][i] la toa do cac diem anh lay tu mang transform_coord
        # se duoc gan bang toa do image ban dau tu orig_coord
        rotated2[transform_coord[1][i],transform_coord[0][i]] = image[orig_coord[1][i], orig_coord[0][i]]


cv2.imshow("rotated2", rotated2)
cv2.waitKey(0)