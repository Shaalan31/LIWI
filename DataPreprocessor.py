import glob
import sift
import common
import h5py
import cv2 as cv
import numpy as np

#IAmDatabase preprocessor
def data_preprocessor():
    SDpoints = np.zeros((1,128))
    for filename in glob.glob('WordsDatabase/*/*/*.png'):
        SDpoints = np.append(SDpoints,sift.getDes(cv.imread(filename)),axis=0)
    with h5py.File('SDpoints.h5', 'w') as hf:
        hf.create_dataset("keypoints-of-Iam", data=SDpoints)


data_preprocessor()