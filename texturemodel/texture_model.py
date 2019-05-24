import glob
import warnings
from itertools import combinations
from sklearn import decomposition
from sklearn import svm
from texturemodel.texture_features import *
from texturemodel.block_segmentation import *

warnings.filterwarnings("ignore", category=RuntimeWarning)

randomState = 1545481387


class TextureWriterIdentification:
    def __init__(self, path_training_set="", path_test_cases=""):
        self.num_features = 256 * 2
        self.all_features = np.asarray([])
        self.all_features_class = np.asarray([])
        self.labels = []
        self.temp = []
        self.num_training_examples = 0
        self.num_blocks_per_class = 0
        self.total_test_cases = 100
        self.pathTrainingSet = path_training_set
        self.pathTestCases = path_test_cases
        self.classifier = svm.SVC(C=10.0, cache_size=200, class_weight=None, coef0=0.0,
                             decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
                             max_iter=-1, probability=True, random_state=1545481387, shrinking=True,
                             tol=0.001, verbose=False)

    def feature_extraction(self, example):
        example = example.astype('uint8')
        example_copy = example.copy()
        local_descriptor = LocalDescriptor()
        feature = []

        feature.extend(np.histogram(local_descriptor.get_lbp_features(example_copy), bins=256)[0])

        feature.extend(np.histogram(np.array(local_descriptor.get_lpq_features(example_copy)), bins=256)[0])

        return np.asarray(feature)

    def fast_test(self,all_features_test,mu,sigma,pca,lang='en'):
        num_testing_examples = all_features_test.shape[0]
        all_features_test = (self.adjust_nan_values(
            np.reshape(all_features_test, (num_testing_examples, self.num_features))) - mu) / sigma
        return np.average(self.classifier.predict_proba(pca.transform(all_features_test)), axis=0).reshape(1, -1)

    def test(self, image, mu, sigma, pca,lang="en"):
        all_features_test = np.asarray([])

        if image.shape[0] > 3500:
            image = cv2.resize(src=image, dsize=(3500, round((3500 / image.shape[1]) * image.shape[0])))

        # image = adjust_rotation(image=image)
        writer_blocks = BlockSegmentation(image,lang).segment()

        num_testing_examples = 0
        for block in writer_blocks:
            example = self.feature_extraction(block)
            all_features_test = np.append(all_features_test, example)
            num_testing_examples += 1

        all_features_test = (self.adjust_nan_values(
            np.reshape(all_features_test, (num_testing_examples, self.num_features))) - mu) / sigma

        # Predict on each line
        # predictions = []
        # for example in all_features_test:
        #     predictions.append(clf.predict(np.asarray(example).reshape(1, -1)))
        # values, counts = np.unique(np.asarray(predictions), return_counts=True)
        # return values[np.argmax(counts)]
        return np.average(self.classifier.predict_proba(pca.transform(all_features_test)),axis=0).reshape(1, -1)

    def training(self, image, class_num):
        image_height = image.shape[0]
        if image_height > 3500:
            image = cv2.resize(src=image, dsize=(3500, round((3500 / image.shape[1]) * image_height)))

        # image = adjust_rotation(image=image)
        writer_blocks = BlockSegmentation(image).segment()

        self.num_blocks_per_class += len(writer_blocks)

        for block in writer_blocks:
            self.all_features_class = np.append(self.all_features_class, self.feature_extraction(block))
            self.labels.append(class_num)
            self.num_training_examples += 1

        return np.reshape(self.all_features_class, (self.num_blocks_per_class, self.num_features))

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
        if len(np.where(deviation == 0)[0]) > 0:
            deviation[np.where(deviation == 0)[0]] = 1
        normalized_X = np.divide(normalized_X, deviation)
        return normalized_X, mean, deviation

    def get_features(self,image,lang="en",h_coeff=None):
        image_height = image.shape[0]
        if image_height > 3500:
            image = cv2.resize(src=image, dsize=(3500, round((3500 / image.shape[1]) * image_height)))

        # image = adjust_rotation(image=image)
        # show_images([image])
        writer_blocks = BlockSegmentation(image,lang,h_coeff=h_coeff).segment()
        num_blocks = len(writer_blocks)

        all_features_class = np.asarray([])
        self.num_blocks_per_class += len(writer_blocks)
        for block in writer_blocks:
            all_features_class = np.append(all_features_class, self.feature_extraction(block))

        return num_blocks,np.reshape(all_features_class, (1, num_blocks * self.num_features))

    def run(self):

        results_array = []
        startClass = 10
        endClass = 20
        classCombinations = combinations(range(startClass, endClass + 1), r=3)

        total_cases = 0
        total_correct = 0

        for classCombination in classCombinations:
            # if 10 in list(classCombination) or 26 in list(classCombination):
            #     continue
            # try:
            self.labels = []
            self.all_features = []
            self.num_training_examples = 0

            for class_number in classCombination:
                self.num_blocks_per_class = 0
                self.all_features_class = np.asarray([])
                for filename in glob.glob(
                        self.pathTrainingSet + str(class_number) + '/*.tif'):
                    print(filename)
                    self.temp = self.training(cv2.imread(filename), class_number)
                self.all_features = np.append(self.all_features,
                                              np.reshape(self.adjust_nan_values(self.temp),
                                                         (1, self.num_blocks_per_class * self.num_features)))

            # Normalization of features
            self.all_features, mu, sigma = self.feature_normalize(
                np.reshape(self.all_features, (self.num_training_examples, self.num_features)))
            pca = decomposition.PCA(n_components=min(self.all_features.shape[0], self.all_features.shape[1]),
                                    svd_solver='full')
            self.all_features = pca.fit_transform(self.all_features)
            self.classifier.fit(self.all_features, self.labels)

            for class_number in classCombination:
                for filename in glob.glob(
                        self.pathTestCases + str(
                            class_number) + '.png'):
                    print(filename)
                    label = class_number
                    prediction = self.test(cv2.imread(filename),mu, sigma, pca)
                    classes = self.classifier.classes_
                    prediction=classes[np.argmax(prediction)]
                    total_cases += 1
                    print(prediction)
                    if prediction == label:
                        total_correct += 1
                    results_array.append(str(prediction) + '\n')
                    print("Accuracy = ", total_correct * 100 / total_cases, " %")

        results_file = open("results.txt", "w+")
        results_file.writelines(results_array)
        results_file.close()

    def fit_classifier(self,all_features,labels):
        self.classifier.fit(all_features, labels)

    def get_classifier_classes(self):
       return self.classifier.classes_

    def get_num_features(self):
        return self.num_features
