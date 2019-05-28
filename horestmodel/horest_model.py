import glob
import warnings
from itertools import combinations
from horestmodel.line_segmentation import *
from horestmodel.horest_features import *
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore", category=RuntimeWarning)

randomState = 1545481387


class HorestWriterIdentification:
    def __init__(self, socket=None, path_training_set="", path_test_cases=""):
        self.num_features = 18
        self.all_features = np.asarray([])
        self.all_features_class = np.asarray([])
        self.labels = []
        self.temp = []
        self.num_training_examples = 0
        self.num_lines_per_class = 0
        self.total_test_cases = 0
        self.right_test_cases = 0
        self.pathTrainingSet = path_training_set
        self.pathTestCases = path_test_cases
        # self.classifier = MLPClassifier(solver='lbfgs', max_iter=1000, alpha=0.046041,
        #                                 hidden_layer_sizes=(22, 18, 15, 12, 7,),
        #                                 random_state=randomState)
        self.classifier = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
        self.socketIO = socket

    def feature_extraction(self, example, image_shape):
        example = example.astype('uint8')
        example_copy = example.copy()

        # Calculate Contours
        _, contours, hierarchy = cv2.findContours(example_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hierarchy = hierarchy[0]
        contours = np.asarray(contours)

        horest_features = HorestFeatures(example, contours, hierarchy, socket=self.socketIO)
        feature = []

        # feature 1, Angles Histogram
        feature.extend(horest_features.angles_histogram())

        # feature 2, Blobs Detection
        feature.extend(horest_features.blob_threaded())

        # feature 3, Connected Components
        feature.extend(horest_features.connected_components(image_shape))

        # feature 4, Disk Fractal
        feature.extend(horest_features.disk_fractal())

        return np.asarray(feature)

    def test(self, image, mu, sigma):
        print("ay haga")
        all_features_test = np.asarray([])

        if image.shape[0] > 3500:
            image = cv2.resize(src=image, dsize=(3500, round((3500 / image.shape[1]) * image.shape[0])))

        # image = adjust_rotation(image=image)
        # show_images([image])
        writer_lines = LineSegmentation(image, self.socketIO).segment()

        num_testing_examples = 0
        for line in writer_lines:
            example = self.feature_extraction(line, image.shape)
            all_features_test = np.append(all_features_test, example)
            num_testing_examples += 1

        all_features_test = (self.adjust_nan_values(
            np.reshape(all_features_test, (num_testing_examples, self.num_features))) - mu) / sigma

        return np.average(self.classifier.predict_proba(all_features_test), axis=0).reshape(1, -1)

    def training(self, image, class_num):
        image_height = image.shape[0]
        if image_height > 3500:
            image = cv2.resize(src=image, dsize=(3500, round((3500 / image.shape[1]) * image_height)))

        # image = adjust_rotation(image=image)
        # show_images([image])
        writer_lines = LineSegmentation(image).segment()

        self.num_lines_per_class += len(writer_lines)
        for line in writer_lines:
            self.all_features_class = np.append(self.all_features_class, self.feature_extraction(line, image.shape))
            self.labels.append(class_num)
            self.num_training_examples += 1

        return np.reshape(self.all_features_class, (self.num_lines_per_class, self.num_features))

    def adjust_nan_values(self, writer_features):
        for i in range(self.num_features):
            feature = writer_features[:, i]
            is_nan_mask = np.isnan(feature)
            if len(np.where(np.asarray(is_nan_mask))[0]) == 0:
                continue

            non_nan_indices = np.where(np.logical_not(is_nan_mask))[0]
            nan_indices = np.where(is_nan_mask)[0]

            if len(non_nan_indices) == 0:
                feature_mean = 0
            else:
                feature_mean = np.mean(feature[non_nan_indices])
            writer_features[nan_indices, i] = feature_mean

        return writer_features

    def feature_normalize(self, X):
        mean = np.mean(X, axis=0)
        normalized_X = X - mean
        deviation = np.sqrt(np.var(normalized_X, axis=0))
        normalized_X = np.divide(normalized_X, deviation)
        return normalized_X, mean, deviation

    randomState = 1545481387

    def run(self):

        results_array = []

        start_class = 1
        end_class = 50
        class_combinations = combinations(range(start_class, end_class + 1), r=30)

        total_cases = 0
        total_correct = 0
        for classCombination in class_combinations:
            # if 10 in list(classCombination) or 26 in list(classCombination):
            #     continue
            # try:
            self.labels = []
            self.all_features = []
            self.num_training_examples = 0

            for class_number in classCombination:
                self.num_lines_per_class = 0
                self.all_features_class = np.asarray([])
                for filename in glob.glob(
                        self.pathTrainingSet + str(class_number) + '/*.png'):
                    print(filename)
                    self.temp = self.training(cv2.imread(filename), class_number)
                self.all_features = np.append(self.all_features,
                                              np.reshape(self.adjust_nan_values(self.temp),
                                                         (1, self.num_lines_per_class * self.num_features)))

            # Normalization of features
            self.all_features, mu, sigma = self.feature_normalize(
                np.reshape(self.all_features, (self.num_training_examples, self.num_features)))

            self.classifier.fit(self.all_features, self.labels)

            for class_number in classCombination:
                for filename in glob.glob(
                        self.pathTestCases + str(
                            class_number) + '_*.png'):
                    print(filename)
                    label = class_number
                    prediction, classes = self.test(cv2.imread(filename), mu, sigma)
                    prediction = classes[np.argmax(prediction)]
                    total_cases += 1
                    print(prediction)
                    if prediction == label:
                        total_correct += 1
                    results_array.append(str(prediction) + '\n')
                    print("Accuracy = ", total_correct * 100 / total_cases, " %")
                    break

        results_file = open("results.txt", "w+")
        results_file.writelines(results_array)
        results_file.close()

    def get_features(self, image):
        image_height = image.shape[0]
        if image_height > 3500:
            image = cv2.resize(src=image, dsize=(3500, round((3500 / image.shape[1]) * image_height)))

        # image = adjust_rotation(image=image)
        # show_images([image])
        writer_lines = LineSegmentation(image).segment()
        num_lines = len(writer_lines)

        all_features_class = np.asarray([])
        self.num_lines_per_class += len(writer_lines)
        for line in writer_lines:
            all_features_class = np.append(all_features_class, self.feature_extraction(line, image.shape))

        return num_lines, np.reshape(all_features_class, (1, num_lines * self.num_features))

    def fit_classifier(self, all_features, labels):
        self.classifier.fit(all_features, labels)

    def get_classifier_classes(self):
        return self.classifier.classes_

    def get_num_features(self):
        return self.num_features
