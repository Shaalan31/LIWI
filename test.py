import cv2
import features
import pickle
from word_segmentation import *
from feature_matching import *


code_book = pickle.load( open( "centers.pkl", "rb" ) )
image = cv2.imread('sample3.png')
image = remove_shadow(image)

# extract handwriting from image
top, bottom = extract_text(image)
image = image[top:bottom, :]
cv2.imwrite('image_extract_text.png', image)

# segment words and get its sift descriptors and orientations
sd, so = word_segmentation(image)

SDS_I1 = features.sds(sd, code_book, t=30)
#SOH_I1 = features.soh(so, phi=10)

# D = match(u=SDS_I1, v=SDS_I2, x=SOH_I1, y=SOH_I2, w=0.8)


#
# img = cv2.imread('a01-026u-02-02.png')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
# sift = cv2.xfeatures2d.SIFT_create()
# kp, des = sift.detectAndCompute(gray, None)
# img=cv2.drawKeypoints(gray,kp,outImage=np.array([]))
# cv2.imwrite("new.png",img)
# print( len(kp),'\n \n \n \n lmaaaaaaaaaaaaaaaao \n \n \n',des.shape)
#
#
# x = np.zeros((1,2))
# print(x)
#
# n = np.append(x,[[3,1],[4,2],[4,2]],axis=0)
# print(n,'\n',n.shape)
# n = n[np.where(n != 0.)]
# print(n,'\n',n.shape)
#
# n= n.reshape((int(n.shape[0]/2),2))
# print(n,'\n',n.shape)
#
#
# with h5py.File('SDpoints.h5', 'r') as hf:
#     data = hf['keypoints-of-Iam'][:]
# print(data.shape)