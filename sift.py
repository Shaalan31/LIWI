import cv2
import numpy as np

def getKeypoints(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    img = cv2.drawKeypoints(gray, kp, outImage=np.array([]))
    cv2.imwrite("new.png", img)
    key_points= np.zeros((len(kp),2))
    for i in range(len(kp)):
        key_points[i,0] = kp[i].angle
        key_points[i, 1] = kp[i].octave
    return key_points,des


