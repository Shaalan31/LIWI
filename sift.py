import cv2
import numpy as np

def getKeypoints(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    key_points= np.zeros((len(kp),2))
    for i in range(len(kp)):
        key_points[i,0] = kp[i].angle
        key_points[i, 1] = kp[i].octave
    return key_points,des


def getDes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    return des


