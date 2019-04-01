import cv2
import numpy as np
import h5py

img = cv2.imread('a01-026u-02-02.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(gray, None)
img=cv2.drawKeypoints(gray,kp,outImage=np.array([]))
cv2.imwrite("new.png",img)
print( len(kp),'\n \n \n \n lmaaaaaaaaaaaaaaaao \n \n \n',des.shape)


x = np.zeros((1,2))
print(x)

n = np.append(x,[[3,1],[4,2],[4,2]],axis=0)
print(n,'\n',n.shape)
n = n[np.where(n != 0.)]
print(n,'\n',n.shape)

n= n.reshape((int(n.shape[0]/2),2))
print(n,'\n',n.shape)


with h5py.File('SDpoints.h5', 'r') as hf:
    data = hf['keypoints-of-Iam'][:]
print(data.shape)