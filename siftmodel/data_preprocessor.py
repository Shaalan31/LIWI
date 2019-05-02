import glob
import h5py
import cv2 as cv
import numpy as np
from siftmodel.sift import *
from utils import *


#IAmDatabase preprocessor
def data_preprocessor():
    sift = Sift()
    with open("Output.txt", 'w') as out:
        out.write("")
    batch_num = 0
    SDpoints = np.zeros((1,128))
    for filename in glob.glob('WordsDatabase/*/*/*.png'):
        temp = sift.get_des(cv.imread(filename))
        if temp is not None:
            SDpoints = np.append(SDpoints,temp,axis=0)
        if SDpoints.shape[0] > 40000:
            SDpoints = np.delete(SDpoints, (0), axis=0)
            #SDpoints, _,_ = feature_normalize(SDpoints)
            with h5py.File('Datasets/SDpoints'+str(batch_num) +'.h5', 'w') as hf:
                hf.create_dataset("keypoints-batch", data=SDpoints)

            with open("Output.txt", 'a') as out:
                out.write(str(batch_num) + "   " +filename + "    " + str(SDpoints.shape[0]) + "\n")
            #print(str(batch_num) + "   " +filename + "    " + str(SDpoints.shape[0]) + "\n")
            batch_num += 1
            SDpoints = np.zeros((1, 128))
    with h5py.File('Datasets/SDpoints' + str(batch_num) + '.h5', 'w') as hf:
        hf.create_dataset("keypoints-batch", data=SDpoints)

    with open("Output.txt", 'a') as out:
        out.write(str(batch_num) + "   " + filename + "    " + str(SDpoints.shape[0]) + "\n")
    print(str(batch_num) + "   " + filename + "    " + str(SDpoints.shape[0]) + "\n")
    batch_num += 1



def normalize():
    with open("stats.txt", 'w') as out:
        out.write("")
    SDpoints = np.zeros((1, 128))
    SDpointsFinal = np.zeros((1, 128))
    for x in range(0,244):
        print(x)
        with h5py.File('Datasets/SDpoints'+str(x) +'.h5', 'r') as hf:
            SDpoints = np.append(SDpoints, hf['keypoints-batch'][:],axis=0)
        if(x % 50 == 0):
            SDpoints = np.delete(SDpoints, (0), axis=0)
            SDpointsFinal = np.append(SDpointsFinal,SDpoints,axis=0)
            print(SDpointsFinal.shape)
            SDpoints = np.zeros((1, 128))
    SDpoints = np.delete(SDpoints, (0), axis=0)
    SDpointsFinal = np.append(SDpointsFinal, SDpoints,axis=0)
    SDpoints = np.zeros((1, 128))
    #print(SDpointsFinal.shape)
    SDpointsFinal = np.delete(SDpointsFinal, (0), axis=0)
    #SDpointsFinal, mean,dev = feature_normalize(SDpointsFinal)

    mean = np.mean(SDpointsFinal, axis=0)
    print(SDpointsFinal)
    mean = mean.reshape((1,128))
    print(SDpointsFinal.transpose().shape, SDpointsFinal.shape)
    normalized_X = SDpointsFinal - mean
    print(normalized_X.shape)
    deviation = np.sqrt(np.var(normalized_X, axis=0))
    normalized_X = np.divide(normalized_X, deviation)
    with open("stats.txt", 'a') as out:
        out.write("Mean:  "+str(mean) + " \nDev: " + str(deviation))
    with h5py.File('Datasets/AData.h5', 'w') as hf:
        hf.create_dataset("keypoints-batch", data=normalized_X)



#data_preprocessor()

normalize()