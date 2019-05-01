import cv2
import features
import pickle
import glob
from pathlib import Path
from word_segmentation import *
from feature_matching import *


def tester(name, path):
    image = cv2.imread(name)
    image = remove_shadow(image)

    # extract handwriting from image
    top, bottom = extract_text(image)
    image = image[top:bottom, :]
    cv2.imwrite('image_extract_text.png', image)

    # segment words and get its sift descriptors and orientations
    sd, so = word_segmentation(image, path)

    # calculate SDS and SOH
    SDS = features.sds(sd, code_book, t=1)
    SOH = features.soh(so, phi=36)

    return SDS, SOH


# Train the model
def train(first_class, last_class):

    base_train = 'C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/Samples/'
    
    count = first_class
    SDS_train = []
    SOH_train = []
    print('Training The Model:')
    
    while count <= last_class:
        for filename in glob.glob(base_train + 'Class' + str(count)):
            print('Class' + str(count) + ':')

            for image in glob.glob(filename + '/*.png'):
                name = Path(image).name
                print(name)
    
                SDS, SOH = tester(base_train + 'Class' + str(count) + '/' + name, name)
                SDS_train.append(SDS)
                SOH_train.append(SOH)
    
            count += 1
            
    return SDS_train, SOH_train


# Test the model
def test(first_class, last_class, SDS_train, SOH_train):

    base_test = 'C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/TestCases/'
    total_test_cases = 0
    right_test_cases = 0
    count = first_class
    print('Testing The Model:')
    
    while count <= last_class:
        print('Class' + str(count) + ':')
    
        for filename in glob.glob(base_test + 'testing' + str(count) + '_*.png'):
            name = Path(filename).name
    
            SDS, SOH = tester(base_test + name, name)
    
            distances = []
            for i in range(0, len(SDS_train)):
                D = match(u=SDS, v=SDS_train[i], x=SOH, y=SOH_train[i], w=0.35)
                distances.append(D)
    
            class_numb = int(np.argmin(distances) / 2) + first_class
            print(name + ' , class number: ' +  str(class_numb))
    
            if(class_numb == count):
                right_test_cases += 1
            total_test_cases += 1
    
            accuracy = (right_test_cases / total_test_cases) * 100
            print('Accuracy:  ' + str(accuracy) + '%')
    
            # break
        count += 1


# code book
code_book = pickle.load( open( "centers.pkl", "rb" ) )
# tester('C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/TestCases/testing24_2.png', 'testing24_2.png')

SDS, SOH = train(first_class=61, last_class=90)
test(first_class=61, last_class=90, SDS_train=SDS, SOH_train=SOH)







# img = cv2.imread('a01-026u-02-02.png')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
# code_book = pickle.load( open( "centers.pkl", "rb" ) )
#
# def tester(name):
#     print(name)
#     image = cv2.imread(name)
#     image = remove_shadow(image)
#
#     # extract handwriting from image
#     top, bottom = extract_text(image)
#     image = image[top:bottom, :]
#     cv2.imwrite('image_extract_text.png', image)
#
#     # segment words and get its sift descriptors and orientations
#     sd, so = word_segmentation(image)
#
#     # calculate SDS and SOH
#     SDS_I1 = features.sds(sd, code_book, t=30)
#     SOH_I1 = features.soh(so, phi=36)
#     print(SDS_I1,'\n',SOH_I1)
#     return SDS_I1,SOH_I1
#
# SDS0,SOH0 = tester('a01-000u.png')
# SDS1,SOH1 = tester('a01-000x.png')
# SDST,SOHT = tester('a01-007u.png')
#

#
# #print(SDS_I1.shape,SOH_I1.shape)
# D0 = match(u=SDS0, v=SDST, x=SOH0, y=SOHT, w=0.1)
# D1 = match(u=SDS1, v=SDST, x=SOH1, y=SOHT, w=0.1)
#
# print(D0,D1)
#
# # img = cv2.imread('a01-026u-02-02.png')
# # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# #
# # sift = cv2.xfeatures2d.SIFT_create()
# # kp, des = sift.detectAndCompute(gray, None)
# # img=cv2.drawKeypoints(gray,kp,outImage=np.array([]))
# # cv2.imwrite("new.png",img)
# # print( len(kp),'\n \n \n \n lmaaaaaaaaaaaaaaaao \n \n \n',des.shape)
# #
# #
# # x = np.zeros((1,2))
# # print(x)
# #
# # n = np.append(x,[[3,1],[4,2],[4,2]],axis=0)
# # print(n,'\n',n.shape)
# # n = n[np.where(n != 0.)]
# # print(n,'\n',n.shape)
# #
# # n= n.reshape((int(n.shape[0]/2),2))
# # print(n,'\n',n.shape)
# #
# #
# # with h5py.File('SDpoints.h5', 'r') as hf:
# #     data = hf['keypoints-of-Iam'][:]
# # print(data.shape)
#
# # with h5py.File('SDpoints.h5', 'r') as hf:
# #     data = hf["keypoints-batch12"][:]
# # print(data.shape)
#
#
#
# #import numpy as np
# import matplotlib.pyplot as plt
#
# from neupy import algorithms, utils
#
#
# utils.reproducible()
# plt.style.use('ggplot')
#
#
# X = np.array([
#     [0.1961, 0.9806],
#     [-0.1961, 0.9806],
#     [0.9806, 0.1961],
#     [0.9806, -0.1961],
#     [-0.5812, -0.8137],
#     [-0.8137, -0.5812],
# ])
#
# sofmnet = algorithms.SOFM(
#     n_inputs=2,
#     n_outputs=3,
#
#     step=0.5,
#     show_epoch=20,
#     shuffle_data=True,
#     verbose=True,
#
#     learning_radius=0,
#     features_grid=(3, 1),
# )
#
# plt.plot(X.T[0:1, :], X.T[1:2, :], 'ko')
# sofmnet.train(X, epochs=100)
#
# print("> Start plotting")
# plt.xlim(-1, 1.2)
# plt.ylim(-1, 1.2)
#
# plt.plot(sofmnet.weight[0:1, :], sofmnet.weight[1:2, :], 'bx')
# print(sofmnet.weight)
# print(sofmnet.features_grid)
# for data in X:
#     print( sofmnet.predict(np.reshape(data, (2, 1)).T))

# with h5py.File('SDpoints.h5', 'r') as hf:
#     data = hf['keypoints-of-Iam'][:]
# print(data.shape)

# with h5py.File('SDpoints.h5', 'r') as hf:
#     data = hf["keypoints-batch12"][:]
# print(data.shape)

