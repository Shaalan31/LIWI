import cv2
import numpy as np
import glob
from pathlib import Path
from siftmodel.features import *
from siftmodel.word_segmentation import *
from siftmodel.feature_matching import *
import math
import os
import pickle

class SiftModel:
    def __init__(self, test_classes, code_book,t=1,phi=36,w=0.1,lang='en'):
        self.total_test_cases = 0
        self.right_test_cases=0
        self.w = w
        self.lang = lang
        self.prediction = []
        # self.base_train_SDS = 'C:/Users/omars/Documents/Github/LIWI/Omar/Fast/Samples/SDS/'
        # self.base_train_SOH = 'C:/Users/omars/Documents/Github/LIWI/Omar/Fast/Samples/SOH/'
        # self.base_test_SDS = 'C:/Users/omars/Documents/Github/LIWI/Omar/Fast/TestCases/SDS/'
        # self.base_test_SOH = 'C:/Users/omars/Documents/Github/LIWI/Omar/Fast/TestCases/SOH/'

        # self.base_samples_t = 'C:/Users/omars/Documents/Github/LIWI/Omar/ValidationArabic/Samples/SDS/'
        # self.base_samples_phi = 'C:/Users/omars/Documents/Github/LIWI/Omar/ValidationArabic/Samples/SOH/'
        #
        # self.base_test_t = 'C:/Users/omars/Documents/Github/LIWI/Omar/ValidationArabic/TestCases/SDS/'
        # self.base_test_phi = 'C:/Users/omars/Documents/Github/LIWI/Omar/ValidationArabic/TestCases/SOH/'


        self.base_samples_t = 'C:/Users/omars/Documents/Github/LIWI/Omar/Validation/Samples/SDS/'
        self.base_samples_phi = 'C:/Users/omars/Documents/Github/LIWI/Omar/Validation/Samples/SOH/'

        self.base_test_t = 'C:/Users/omars/Documents/Github/LIWI/Omar/Validation/TestCases/SDS/'
        self.base_test_phi = 'C:/Users/omars/Documents/Github/LIWI/Omar/Validation/TestCases/SOH/'

        self.str_t = str(t)
        self.str_phi = str(phi)

        while len(self.str_t) < 3:
            self.str_t = '0' + self.str_t

        while len(self.str_phi) < 3:
            self.str_phi = '0' + self.str_phi

        self.base_samples_t = self.base_samples_t+self.str_t+'/'
        self.base_test_t = self.base_test_t + self.str_t + '/'

        self.base_samples_phi = self.base_samples_phi + self.str_phi + '/'
        self.base_test_phi = self.base_test_phi + self.str_phi + '/'

        self.test_class = test_classes
        self.code_book = code_book
        # create needed objects
        self.thesis = np.zeros(1)
        self.accuracy = None

    def set_code_book(self, lang):
        if lang == 'en':
            fn = os.path.join(os.path.dirname(__file__), 'centers.pkl')
        elif lang == 'ar':
            fn = os.path.join(os.path.dirname(__file__), 'centers_KHATT.pkl')
        self.code_book = pickle.load(open(fn, "rb"))

    def get_features(self, name, pathSOH,pathSDS):

        SDS = np.genfromtxt(pathSDS, delimiter=",")
        SOH = np.genfromtxt(pathSOH, delimiter=",")

        return SDS, SOH

    # Train the model
    def train(self):


        SDS_train = []
        SOH_train = []
        # print('Training The Model:')

        # print('Get SDS')
        for count in  self.test_class:
            oba = True
            for filename in glob.glob(self.base_samples_t + 'Class' + str(count)):
                oba = False
                # print(count)
                # print('Class' + str(count) + ':')

                for image in glob.glob(filename + '/*.jpg'):
                    name = Path(image).name
                    # print(name)
                    SDS = np.genfromtxt(image, delimiter=",")
                    SDS = SDS.reshape((1,SDS.shape[0]))
                    SDS_train.append(SDS)

            for filename in glob.glob(self.base_samples_phi + 'Class' + str(count)):
                # print('Class' + str(count) + ':')

                for image in glob.glob(filename + '/*.jpg'):
                    name = Path(image).name
                    # print(name)
                    SOH = np.genfromtxt(image, delimiter=",")
                    SOH = SOH.reshape((1,SOH.shape[0]))
                    SOH_train.append(SOH)
            if oba:
                print(count)
        # print(SDS_train.shape)
        return SDS_train, SOH_train

    # Test the model
    def test(self, SDS_train, SOH_train):


        # print('Testing The Model:')

        matching = FeatureMatching()
        for count in self.test_class:
            # print('Class' + str(count) + ':')

            for filename in glob.glob(self.base_test_t + 'testing' + str(count) + '_*.csv'):
                # print(filename)
                name = Path(filename).name

                SDS = np.genfromtxt(filename, delimiter=",")
                SDS = SDS.reshape((1,SDS.shape[0]))
                SOH_filename = filename.replace('SDS/'+self.str_t,'SOH/'+self.str_phi)
                SOH = np.genfromtxt(SOH_filename, delimiter=",")
                SOH = SOH.reshape((1,SOH.shape[0]))

                # Feature Matching and Fusion
                manhattan = []
                chi_square = []
                for i in range(0, len(SDS_train)):
                    D1, D2 = matching.calculate_distances(u=SDS, v=SDS_train[i], x=SOH, y=SOH_train[i])
                    manhattan.append(D1)
                    chi_square.append(D2)
                prediction = matching.match(manhattan, chi_square, w=self.w)
                if self.lang=='en':
                    class_numb = self.test_class[ math.floor(prediction / 1)]
                else:
                    class_numb = self.test_class[math.floor(prediction / 3)]
                # print(name + ' , class number: ' + str(class_numb))
                self.prediction.append(class_numb)
                # Calculate accuracy
                if (class_numb == count):
                    self.right_test_cases += 1
                self.total_test_cases += 1

                accuracy = (self.right_test_cases / self.total_test_cases) * 100
                self.thesis = np.append(self.thesis,accuracy)
                # print('Accuracy:  ' + str(accuracy) + '%')

            # break
        self.accuracy = np.array([[self.right_test_cases], [self.total_test_cases]])

    def run(self):
        SDS, SOH = self.train()
        self.test(SDS_train=SDS, SOH_train=SOH)
