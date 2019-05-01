from segmentation import *
# from AnglesHistogram import *
# from BlobsDetection import *
# from ConnectedComponents import *
# from DiskFractal import *
# from AdjustRotation import *
import glob
import warnings
import random
from sklearn.neural_network import MLPClassifier
from itertools import combinations

warnings.filterwarnings("ignore", category=RuntimeWarning)


def feature_extraction(example, image_shape):
    example = example.astype('uint8')
    example_copy = example.copy()

    feature = []

    lbp = local_binary_pattern(example_copy, P=8, R=1)

    feature.extend(np.histogram(lbp, bins=256)[0])

    return np.asarray(feature)


def test(image, clf, mu, sigma):
    all_features_test = np.asarray([])

    if image.shape[0] > 3500:
        image = cv2.resize(src=image, dsize=(3500, round((3500 / image.shape[1]) * image.shape[0])))

    # image = adjust_rotation(image=image)
    # show_images([image], ["rotation"])
    writer_blocks = segment(image)

    num_testing_examples = 0
    for block in writer_blocks:
        example = feature_extraction(block, image.shape)
        all_features_test = np.append(all_features_test, example)
        num_testing_examples += 1

    all_features_test = (adjustNaNValues(
        np.reshape(all_features_test, (num_testing_examples, num_features))) - mu) / sigma

    # Predict on each line
    # predictions = []
    # for example in all_features_test:
    #     predictions.append(clf.predict(np.asarray(example).reshape(1, -1)))
    # values, counts = np.unique(np.asarray(predictions), return_counts=True)
    # return values[np.argmax(counts)]
    return clf.predict(np.average(all_features_test, axis=0).reshape(1, -1))


def training(image, class_num):
    global all_features
    global all_features_class
    global labels
    global num_blocks_per_class
    global num_training_examples

    image_height = image.shape[0]
    if image_height > 3500:
        image = cv2.resize(src=image, dsize=(3500, round((3500 / image.shape[1]) * image_height)))

    # image = adjust_rotation(image=image)
    writer_blocks = segment(image)

    num_blocks_per_class += len(writer_blocks)

    for block in writer_blocks:
        all_features_class = np.append(all_features_class, feature_extraction(block, image.shape))
        labels.append(class_num)
        num_training_examples += 1

    return np.reshape(all_features_class, (num_blocks_per_class, num_features))


def adjustNaNValues(writer_features):
    for i in range(num_features):
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


def featureNormalize(X):
    mean = np.mean(X, axis=0)
    normalized_X = X - mean
    deviation = np.sqrt(np.var(normalized_X, axis=0))
    if len(np.where(deviation == 0)[0]) > 0:
        deviation[np.where(deviation == 0)[0]] = 1
    normalized_X = np.divide(normalized_X, deviation)
    return normalized_X, mean, deviation


randomState = 1545481387


def reading_test_cases():
    global all_features
    global temp
    global num_training_examples
    global labels
    global num_blocks_per_class
    global all_features_class
    results_array = []

    label = 0
    startClass = 1
    endClass = 100
    classCombinations = combinations(range(startClass, endClass + 1), r=5)

    total_cases = 0
    total_correct = 0
    for classCombination in classCombinations:
        if 10 in list(classCombination) or 26 in list(classCombination):
            continue
        # try:
        labels = []
        all_features = []
        num_training_examples = 0

        for class_number in classCombination:
            num_blocks_per_class = 0
            all_features_class = np.asarray([])
            for filename in glob.glob('D:/Uni/Graduation Project/Samples/Class' + str(class_number) + '/*.tif'):
                print(filename)
                temp = training(cv2.imread(filename), class_number)
            all_features = np.append(all_features,
                                     np.reshape(adjustNaNValues(temp), (1, num_blocks_per_class * num_features)))

        # Normalization of features
        all_features, mu, sigma = featureNormalize(np.reshape(all_features, (num_training_examples, num_features)))
        classifier.fit(all_features, labels)

        for class_number in classCombination:
            for filename in glob.glob('D:/Uni/Graduation Project/TestCases/testing' + str(class_number) + '.png'):
                print(filename)
                label = int(filename[len(filename) - 5])
                prediction = test(cv2.imread(filename), classifier, mu, sigma)
                total_cases += 1
                print(prediction[0])
                if prediction[0] == class_number:
                    total_correct += 1
                results_array.append(str(prediction[0]) + '\n')
                print("Accuracy = ", total_correct * 100 / total_cases, " %")
    # except:
    #     print("EXCEPTIONN!!!")
    #     prediction = str(random.randint(1, 3))
    #     total_cases += 1
    #     if prediction == label:
    #         total_correct += 1
    #     results_array.append(str(prediction) + '\n')
    #     print("Accuracy = ", total_correct * 100 / total_cases, " %")

    results_file = open("results.txt", "w+")
    results_file.writelines(results_array)
    results_file.close()


# Global Variables
all_features = np.asarray([])
all_features_class = np.asarray([])
labels = []
temp = []
num_training_examples = 0
num_features = 256
num_blocks_per_class = 0
total_test_cases = 100

classifier = MLPClassifier(solver='lbfgs', max_iter=30000, alpha=0.046041, hidden_layer_sizes=(22, 18, 15, 12, 7,),
                           random_state=randomState)
reading_test_cases()
