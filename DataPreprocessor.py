import glob
import sift
import common
import h5py
import cv2 as cv
import numpy as np




#IAmDatabase preprocessor
def data_preprocessor():
    with open("Output.txt", 'w') as out:
        out.write("")
    batch_num = 0
    SDpoints = np.zeros((1,128))
    for filename in glob.glob('WordsDatabase/*/*/*.png'):
        temp = sift.getDes(cv.imread(filename))
        if temp is not None:
            SDpoints = np.append(SDpoints,temp,axis=0)
        else:
            print(SDpoints.shape)
        if SDpoints.shape[0] > 1000000:
            SDpoints = np.delete(SDpoints,0,0)
            SDpoints, _,_ = common.feature_normalize(SDpoints)
            with h5py.File('Datasets/SDpoints'+str(batch_num) +'.h5', 'w') as hf:
                hf.create_dataset("keypoints-batch", data=SDpoints)

            with open("Output.txt", 'a') as out:
                out.write(str(batch_num) + "   " +filename + "    " + str(SDpoints.shape[0]) + "\n")
            print(str(batch_num) + "   " +filename + "    " + str(SDpoints.shape[0]) + "\n")
            batch_num += 1
            SDpoints = np.zeros((1, 128))

data_preprocessor()