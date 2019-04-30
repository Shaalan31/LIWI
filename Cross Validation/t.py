#cross validatopn to find the best t value that is used in SDS
import cv2
import features
import pickle
from word_segmentation import *
import numpy as  np
import matplotlib as plot
import pickle
from feature_matching import *
import glob

code_book = pickle.load( open( "centers.pkl", "rb" ) )

def tester(name):
    print(name)
    image = cv2.imread(name)
    image = remove_shadow(image)

    # extract handwriting from image
    top, bottom = extract_text(image)
    image = image[top:bottom, :]
    cv2.imwrite('image_extract_text.png', image)

    # segment words and get its sift descriptors and orientations
    sd, so = word_segmentation(image)

    # calculate SDS and SOH
    SDS_I1 = features.sds(sd, code_book, t=30)
    SOH_I1 = features.soh(so, phi=36)
    print(SDS_I1,'\n',SOH_I1)
    return SDS_I1,SOH_I1

SDS0,SOH0 = tester('a01-000u.png')
SDS1,SOH1 = tester('a01-000x.png')
SDST,SOHT = tester('a01-007u.png')


#print(SDS_I1.shape,SOH_I1.shape)
D0 = match(u=SDS0, v=SDST, x=SOH0, y=SOHT, w=0.1)
D1 = match(u=SDS1, v=SDST, x=SOH1, y=SOHT, w=0.1)

print(D0,D1)

#t from 1 to 300 step 10
def find_opt_t(classes=3,testcases=100):

    num_of_classes=classes
    num_of_testcases=testcases
    test_case=0
    class_num = 1

    for t in range(1,300,10):
        while test_case < testcases:

            for filename in glob.glob('C:/Users/omars/Documents/Github/LIWI/Omar/Samples/Class'+str(class_num)+'/'+ '*.png'):
                print(filename)
            class_num +=1
            break
    return


find_opt_t()

# u: SDS of I1 (first image)
# v: SDS of I2 (second image)
def matchSDS(u, v):
    # Manhattan distance to measure the dissimilarity between two SDSs u and v
    D = np.sum(np.abs(u - v))
    return D

