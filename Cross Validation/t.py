#cross validatopn to find the best t value that is used in SDS
import cv2
import features
import pickle
from word_segmentation import *
import numpy as  np
import matplotlib.pyplot as plt
import pickle
from feature_matching import *
import glob

code_book = pickle.load( open( "C:/Users/omars/Documents/Github/LIWI/centers.pkl", "rb" ) )

#t from 1 to 300 step 10

# u: SDS of I1 (first image)
# v: SDS of I2 (second image)
def matchSDS(u, v,real_class):
    # Manhattan distance to measure the dissimilarity between two SDSs u and v
    D = np.sum(np.abs(u - v),axis=1)
    print(np.argmin(D)/2)
    return int(real_class  == int(np.argmin(D)/2))



def find_opt_t(classes=3,testcases=21):
    #test_case=0

    accuracy = np.zeros((10))
    print(accuracy)
    xaxis = np.zeros((10))


    for t in range(1,300,30):
        xaxis[int(t/30)] = t
        class_num = 1
        passed_cases = 0
        total_cases=0
        while class_num < testcases:
            counter = 0
            SDS = np.zeros((2 * classes, 300))
            test_case_num = class_num
            for x in range(classes):
                for filename in glob.glob('C:/Users/omars/Documents/Github/LIWI/Omar/Samples/Class'+str(class_num)+'/'+ '*.png'):
                    name = filename
                    print(filename)
                    image = cv2.imread(name)
                    image = remove_shadow(image)

                    # extract handwriting from image
                    top, bottom = extract_text(image)
                    image = image[top:bottom, :]
                    cv2.imwrite('image_extract_text.png', image)

                    # segment words and get its sift descriptors and orientations
                    sd, so = word_segmentation(image)

                    # calculate SDS and SOH
                    SDS[counter] = features.sds(sd, code_book, t=t)
                    counter +=1
                class_num +=1

            class_count = 0
            for x in range(test_case_num, test_case_num + classes):

                for filename in glob.glob('C:/Users/omars/Documents/Github/LIWI/Omar/TestCases/testing' + str(test_case_num)  + '_*.png'):
                    name = filename
                    print(filename)
                    image = cv2.imread(name)
                    image = remove_shadow(image)

                    # extract handwriting from image
                    top, bottom = extract_text(image)
                    image = image[top:bottom, :]
                    #cv2.imwrite('image_extract_text.png', image)

                    # segment words and get its sift descriptors and orientations
                    sd, so = word_segmentation(image)

                    # calculate SDS and SOH
                    SDS_case = features.sds(sd, code_book, t=t)
                    passed_cases += matchSDS(SDS,SDS_case,class_count)
                    total_cases += 1
                    print('accuracy: ', passed_cases / total_cases)
                test_case_num +=1
                class_count +=1



        print('accuracy: ',passed_cases/total_cases)
        accuracy[int(t/30)] = passed_cases/total_cases
        print(accuracy.shape)
    print(accuracy)
    plt.plot(xaxis, accuracy)
    plt.show()
    plt.savefig('t.png')
    return



find_opt_t()

