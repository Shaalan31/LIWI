import cv2
import numpy as np
import glob
from pathlib import Path
from siftmodel.features import *
from siftmodel.word_segmentation import *
from siftmodel.feature_matching import *


class SiftModel:
    def __init__(self, test_classes, code_book):
        self.base_train_SDS = 'C:/Users/omars/Documents/Github/LIWI/Omar/Fast/Samples/SDS/'
        self.base_train_SOH = 'C:/Users/omars/Documents/Github/LIWI/Omar/Fast/Samples/SOH/'
        self.base_test_SDS = 'C:/Users/omars/Documents/Github/LIWI/Omar/Fast/TestCases/SDS/'
        self.base_test_SOH = 'C:/Users/omars/Documents/Github/LIWI/Omar/Fast/TestCases/SOH/'
        self.test_class = test_classes
        self.code_book = code_book
        # create needed objects

        self.accuracy = None

    def get_features(self, name, pathSOH,pathSDS):

        SDS = np.genfromtxt(pathSDS, delimiter=",")
        SOH = np.genfromtxt(pathSOH, delimiter=",")

        return SDS, SOH

    # Train the model
    def train(self):


        SDS_train = []
        SOH_train = []
        print('Training The Model:')

        print('Get SDS')
        for count in  self.test_class:
            for filename in glob.glob(self.base_train_SDS + 'Class' + str(count)):
                print('Class' + str(count) + ':')

                for image in glob.glob(filename + '/*.csv'):
                    name = Path(image).name
                    print(name)
                    SDS = np.genfromtxt(image, delimiter=",")
                    SDS = SDS.reshape((1,SDS.shape[0]))
                    SDS_train.append(SDS)

            for filename in glob.glob(self.base_train_SOH + 'Class' + str(count)):
                print('Class' + str(count) + ':')

                for image in glob.glob(filename + '/*.csv'):
                    name = Path(image).name
                    print(name)
                    SOH = np.genfromtxt(image, delimiter=",")
                    SOH = SOH.reshape((1,SOH.shape[0]))
                    SOH_train.append(SOH)

        return SDS_train, SOH_train

    # Test the model
    def test(self, SDS_train, SOH_train):

        total_test_cases = 0
        right_test_cases = 0
        print('Testing The Model:')

        matching = FeatureMatching()
        for count in self.test_class:
            print('Class' + str(count) + ':')

            for filename in glob.glob(self.base_test_SDS + 'testing' + str(count) + '_*.csv'):
                name = Path(filename).name

                SDS = np.genfromtxt(filename, delimiter=",")
                SDS = SDS.reshape((1,SDS.shape[0]))
                SOH_filename = filename.replace('SDS','SOH')
                SOH = np.genfromtxt(SOH_filename, delimiter=",")
                SOH = SOH.reshape((1,SOH.shape[0]))

                # Feature Matching and Fusion
                manhattan = []
                chi_square = []
                for i in range(0, len(SDS_train)):
                    D1, D2 = matching.calculate_distances(u=SDS, v=SDS_train[i], x=SOH, y=SOH_train[i])
                    manhattan.append(D1)
                    chi_square.append(D2)
                prediction = matching.match(manhattan, chi_square, w=0.75)

                class_numb = self.test_class[ int(prediction / 2)-1]
                print(name + ' , class number: ' + str(class_numb))

                # Calculate accuracy
                if (class_numb == count):
                    right_test_cases += 1
                total_test_cases += 1

                accuracy = (right_test_cases / total_test_cases) * 100

                print('Accuracy:  ' + str(accuracy) + '%')

            # break
        self.accuracy = np.array([[right_test_cases], [total_test_cases]])

    def run(self):
        SDS, SOH = self.train()
        self.test(SDS_train=SDS, SOH_train=SOH)
