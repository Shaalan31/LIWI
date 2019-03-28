import cv2
import numpy as np

img = cv2.imread('a01-026u-02-02.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(gray, None)
img=cv2.drawKeypoints(gray,kp,outImage=np.array([]))
cv2.imwrite("new.png",img)
print( kp[0].response,'\n \n \n \n lmaaaaaaaaaaaaaaaao \n \n \n',des)