import cv2
import glob
from pathlib import Path
from siftmodel.features import *
from siftmodel.word_segmentation import *
from siftmodel.feature_matching import *
import pickle
import os


class SiftModel:
    def __init__(self, first_class=1, last_class=1):
        self.base_train = 'C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/Samples/'
        self.base_test = 'D:/Uni/Graduation Project/All Test Cases/IAMJPG/TestCases/'
        self.first_class = first_class
        self.last_class = last_class
        self.code_book = None
        # create needed objects
        self.preprocess = Preprocessing()
        self.features = FeaturesExtraction()
        self.accuracy = None

    def get_features(self, name, lang="en", image=None, path="",t=1,phi=36):

        if image is None:
            image = cv2.imread(path)
        image = self.preprocess.remove_shadow(image)

        # extract handwriting from image
        if lang == "en":
            top, bottom = self.preprocess.extract_text(image)
            image = image[top:bottom, :]
        else:
            image= cv2.copyMakeBorder(image,10,10,10,10,cv2.BORDER_CONSTANT,value=[255,255,255])

        cv2.imwrite('C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/image_extract_text.png', image)

        # segment words and get its sift descriptors and orientations
        sd, so = WordSegmentation(lang).word_segmentation(image, name)

        # calculate SDS and SOH
        SDS = self.features.sds(sd, self.code_book, t=t)
        SOH = self.features.soh(so, phi=phi)

        return SDS, SOH

    # Train the model
    def train(self):

        count = self.first_class
        SDS_train = []
        SOH_train = []
        print('Training The Model:')

        while count <= self.last_class:
            for filename in glob.glob(self.base_train + 'Class' + str(count)):
                print('Class' + str(count) + ':')

                for image in glob.glob(filename + '/*.png'):
                    name = Path(image).name
                    print(name)

                    # Feature Extraction
                    SDS, SOH = self.get_features(name, path=self.base_train + 'Class' + str(count) + '/' + name)
                    SDS_train.append(SDS)
                    SOH_train.append(SOH)

                count += 1

        return SDS_train, SOH_train

    # Test the model
    def test(self, SDS_train, SOH_train):

        total_test_cases = 0
        right_test_cases = 0
        count = self.first_class
        print('Testing The Model:')

        matching = FeatureMatching()
        while count <= self.last_class:
            print('Class' + str(count) + ':')

            for filename in glob.glob(self.base_test + 'testing' + str(count) + '_*.png'):
                name = Path(filename).name

                # Feature Extraction
                SDS, SOH = self.get_features(name, path=self.base_test + name)

                # Feature Matching and Fusion
                manhattan = []
                chi_square = []
                for i in range(0, len(SDS_train)):
                    D1, D2 = matching.calculate_distances(u=SDS, v=SDS_train[i], x=SOH, y=SOH_train[i])
                    manhattan.append(D1)
                    chi_square.append(D2)
                prediction = matching.match(manhattan, chi_square, w=0.75)

                class_numb = int(prediction / 2) + self.first_class
                print(name + ' , class number: ' + str(class_numb))

                # Calculate accuracy
                if (class_numb == count):
                    right_test_cases += 1
                total_test_cases += 1

                accuracy = (right_test_cases / total_test_cases) * 100

                print('Accuracy:  ' + str(accuracy) + '%')

                # break
            count += 1
        self.accuracy = np.array([[right_test_cases], [total_test_cases]])

    def predict(self, SDS_train, SOH_train, testing_image, name, lang="en"):
        matching = FeatureMatching()
        # Feature Extraction
        SDS, SOH = self.get_features(name, image=testing_image,lang=lang)

        # Feature Matching and Fusion
        manhattan = []
        chi_square = []
        for i in range(0, len(SDS_train)):
            D1, D2 = matching.calculate_distances(u=SDS, v=SDS_train[i], x=SOH, y=SOH_train[i])
            manhattan.append(D1)
            chi_square.append(D2)

        if lang == "en":
            prediction = matching.match2(manhattan, chi_square, w=0.75)
        else:
            prediction = matching.match2(manhattan, chi_square, w=0.25)

        return prediction

    def run(self):
        SDS, SOH = self.train()
        self.test(SDS_train=SDS, SOH_train=SOH)

    def set_code_book(self, lang):
        if lang == 'en':
            fn = os.path.join(os.path.dirname(__file__), 'centers.pkl')
        elif lang == 'ar':
            fn = os.path.join(os.path.dirname(__file__), 'centers_KHATT.pkl')
        self.code_book = pickle.load(open(fn, "rb"))
