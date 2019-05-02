import cv2
from siftmodel.features import *
from siftmodel.word_segmentation import *
from siftmodel.feature_matching import *
import numpy as np
import pickle
from pathlib import Path
import glob
import os, errno



code_book = pickle.load( open( "centers.pkl", "rb" ) )
base_train = 'C:/Users/omars/Documents/Github/LIWI/Omar/Samples/'
base_test = 'C:/Users/omars/Documents/Github/LIWI/Omar/TestCases/'
base_train_SDS = 'C:/Users/omars/Documents/Github/LIWI/Omar/Fast/Samples/SDS/'
base_train_SOH = 'C:/Users/omars/Documents/Github/LIWI/Omar/Fast/Samples/SOH/'
base_test_SDS = 'C:/Users/omars/Documents/Github/LIWI/Omar/Fast/TestCases/SDS/'
base_test_SOH = 'C:/Users/omars/Documents/Github/LIWI/Omar/Fast/TestCases/SOH/'
def get_features(name, path):
    segmentation = WordSegmentation()
    preprocess = Preprocessing()
    features = FeaturesExtraction()

    image = cv2.imread(name)
    image = preprocess.remove_shadow(image)

    # extract handwriting from image
    top, bottom = preprocess.extract_text(image)
    image = image[top:bottom, :]
    # cv2.imwrite('image_extract_text.png', image)

    # segment words and get its sift descriptors and orientations
    sd, so = segmentation.word_segmentation(image, path)

    # calculate SDS and SOH
    SDS = features.sds(sd, code_book, t=1)
    SOH = features.soh(so, phi=36)

    return SDS, SOH


def generate_train():
    for count  in range (1,159):
        for filename in glob.glob(base_train + 'Class' + str(count)):
            print('Class' + str(count) + ':')

            for image in glob.glob(filename + '/*.png'):
                name = Path(image).name
                print(name)

                SDS, SOH = get_features(base_train + 'Class' + str(count) + '/' + name, name)
                name = name.replace('png','csv')
                print(SDS.shape)
                print(SOH.shape)



                np.savetxt(base_train_SDS +"Class"+str(count)+"/"+ name, SDS, delimiter=",")
                np.savetxt(base_train_SOH +"Class"+str(count)+"/"+ name, SOH, delimiter=",")




def generate_test():
    for count in range (1,159):
        print('Class' + str(count) + ':')

        for filename in glob.glob(base_test + 'testing' + str(count) + '_*.png'):
            name = Path(filename).name
            SDS, SOH = get_features(base_test + name, name)

            name = name.replace('png', 'csv')
            np.savetxt(base_test_SDS + name, SDS, delimiter=",")
            np.savetxt(base_test_SOH + name, SOH, delimiter=",")



generate_test()