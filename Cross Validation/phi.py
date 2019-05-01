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
def matchSOH(x, ys,real_class,num_of_classes):

    # Chi-Square distance to measure the dissimilarity between SOH x and y
    ChiDistance = np.zeros(num_of_classes*2)
    idx = 0
    for y in ys:
        if (x.shape[1] != y.shape[1]):
            if (x.shape[1] < y.shape[1]):
                padding = np.zeros((x.shape[0], (y.shape[1] - x.shape[1])))
                x = np.append(x, padding,axis=1)
            else:
                padding = np.zeros((y.shape[0], (x.shape[1] - y.shape[1])))
                y = np.append(y, padding,axis=1)

        ChiDistance[idx] = np.sum(np.square(x - y) / (x + y + 1e-16))
        idx += 1

    return int(real_class  == int(np.argmin(ChiDistance)/2))



def find_opt_phi(classes=3,testcases=21):
    test_case=0

    accuracy = np.zeros((30))
    print(accuracy)
    xaxis = np.zeros((30))


    for phi in range(1,360,180):
        test_case = 0
        xaxis[int(phi/12)] = phi
        class_num = 1
        passed_cases = 0
        total_cases=0
        while test_case < testcases:
            counter = 0
            SOH_train = []
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
                    SOH_train.append(features.soh(so, phi=phi))
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
                    SOH = features.soh(so, phi=phi)



                    passed_cases += matchSOH(SOH,SOH_train,class_count,classes)
                    total_cases += 1
                    print('accuracy: ', passed_cases / total_cases)
                test_case_num +=1
                test_case +=1
                class_count +=1


        print('accuracy: ',passed_cases/total_cases)
        accuracy[int(phi/12)] = passed_cases/total_cases
        print(accuracy.shape)
    print(accuracy)
    plt.plot(xaxis, accuracy)
    plt.show()
    plt.savefig('phi.png')
    return



find_opt_phi()

