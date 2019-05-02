import cv2
from siftmodel.features import *
from siftmodel.word_segmentation import *
from siftmodel.feature_matching import *
import numpy as np
import pickle

code_book = pickle.load( open( "siftmodel/centers.pkl", "rb" ) )
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

for x  in range (1,159):
    for