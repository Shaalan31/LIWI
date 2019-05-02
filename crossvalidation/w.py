#test Samar
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
from siftmodel.features import *
from siftmodel.word_segmentation import *

code_book = pickle.load( open( "C:/Users/omars/Documents/Github/LIWI/centers.pkl", "rb" ) )

#ttest kaza mara 3ala kaza w mo5talefa mara wahdaaa (sweep 3al testcases msh 3al w)

# u: SDS of I1 (first image)
# v: SDS of Training (second image)
# x: SOH of I1 (first image)
# y: SOH of Training (second image)
# w: array of probabilities
# real_class: real answer
def match(u, v, x, ys, w,real_class,num_of_classes):

    # Manhattan distance to measure the dissimilarity between two SDSs u and v
    D1 = np.sum(np.abs(u - v), axis=1)
    # Chi-Square distance to measurethe dissimilarity between SOH x and y
    ChiDistance = np.zeros(num_of_classes * 2)
    idx = 0
    for y in ys:
        if (x.shape[1] != y.shape[1]):
            if (x.shape[1] < y.shape[1]):
                padding = np.zeros((x.shape[0], (y.shape[1] - x.shape[1])))
                x = np.append(x, padding, axis=1)
            else:
                padding = np.zeros((y.shape[0], (x.shape[1] - y.shape[1])))
                y = np.append(y, padding, axis=1)

        ChiDistance[idx] = np.sum(np.square(x - y) / (x + y + 1e-16))
        idx += 1

    D2 = ChiDistance

    distances = np.array([D1,D2])
    distances = distances/np.max(distances,axis=0)


    # D new distance to measure the dissimilarity between I1 and I2
#    D = w * distances[0] + (1 - w) * distances[1]
    passed = np.zeros((9))
    idx = 0
    for prob in w:
        D = prob * distances[0,:] + (1 - prob) * distances[1,:]
        passed[idx] = int(real_class  == int(np.argmin(D)/2))
        idx += 1
    return passed


def find_opt_w(classes=3,testcases=159,t=25,phi=36):
    preprocess = Preprocessing()
    segmentation = WordSegmentation()
    features = FeaturesExtraction()

    test_case=0
    accuracy = np.zeros((9))
    print(accuracy)
    xaxis = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])



    class_num = 2
    passed_cases = np.zeros((9))
    total_cases=0
    while test_case < testcases:
        counter = 0
        SDS = np.zeros((2 * classes, 300))
        SOH_train = []
        test_case_num = class_num
        for x in range(classes):
            for filename in glob.glob('C:/Users/omars/Documents/Github/LIWI/Omar/Samples/Class'+str(class_num)+'/'+ '*.png'):
                name = filename
                print(filename)
                image = cv2.imread(name)
                image = preprocess.remove_shadow(image)

                # extract handwriting from image
                top, bottom = preprocess.extract_text(image)
                image = image[top:bottom, :]
                cv2.imwrite('image_extract_text.png', image)

                # segment words and get its sift descriptors and orientations
                sd, so = segmentation.word_segmentation(image)

                # calculate SDS and SOH
                SDS[counter] = features.sds(sd, code_book, t=t)
                SOH_train.append(features.soh(so, phi=phi))
                counter +=1
            class_num +=1

        class_count = 0
        for x in range(test_case_num, test_case_num + classes):

            for filename in glob.glob('C:/Users/omars/Documents/Github/LIWI/Omar/TestCases/testing' + str(test_case_num)  + '_*.png'):
                name = filename
                print(filename)
                image = cv2.imread(name)
                image = preprocess.remove_shadow(image)

                # extract handwriting from image
                top, bottom = preprocess.extract_text(image)
                image = image[top:bottom, :]
                #cv2.imwrite('image_extract_text.png', image)

                # segment words and get its sift descriptors and orientations
                sd, so = segmentation.word_segmentation(image)

                # calculate SDS and SOH
                SDS_case = features.sds(sd, code_book, t=t)
                SOH = features.soh(so, phi=phi)
                passed_cases += match(SDS,SDS_case,SOH,SOH_train,xaxis,class_count,classes)
                total_cases += 1
                print('accuracy: ', passed_cases / total_cases)
            test_case_num +=1
            class_count +=1
        print('accuracy: ',passed_cases/total_cases)
        #accuracy[int(t/100)] = passed_cases/total_cases
        print(accuracy.shape)
    print(accuracy)
    accuracy = passed_cases/total_cases
    plt.plot(xaxis, accuracy)
    plt.show()
    return

find_opt_w()