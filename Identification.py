from segmentation import *
# from AnglesHistogram import *
# from BlobsDetection import *
# from ConnectedComponents import *
# from DiskFractal import *
import glob
import warnings
import time
import random
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore", category=RuntimeWarning)

#
# def feature_extraction(example, image_shape):
#     example = example.astype('uint8')
#     example_copy = example.copy()
#
#     feature = []
#
#     # feature 1, Angles Histogram
#     feature.extend(AnglesHistogram(example))
#
#     # Calculate Contours
#     _, contours, hierarchy = cv2.findContours(example_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     hierarchy = hierarchy[0]
#     contours = np.asarray(contours)
#
#     # feature 2, Blobs Detection
#     feature.extend(blob_threaded(contours, hierarchy))
#
#     # feature 3, Connected Components
#     feature.extend(ConnectedComponents(contours, hierarchy, example_copy, image_shape))
#
#     # feature 4, Disk Fractal
#     feature.extend(DiskFractal(example_copy))
#
#     return np.asarray(feature)


def test(image, clf, mu, sigma):
    all_features_test = np.asarray([])

    if image.shape[0] > 3500:
        image = cv2.resize(src=image, dsize=(3500, round((3500 / image.shape[1]) * image.shape[0])))

    # image = adjust_rotation(image=image)
    # show_images([image], ["rotation"])
    writerLines = segment(image)

    num_testing_examples = 0
    # for line in writerLines:
    #     example = feature_extraction(line, image.shape)
    #     all_features_test = np.append(all_features_test, example)
    #     num_testing_examples += 1

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
    global num_lines_per_class
    global num_training_examples

    image_height = image.shape[0]
    if image_height > 3500:
        image = cv2.resize(src=image, dsize=(3500, round((3500 / image.shape[1]) * image_height)))

    # image = adjust_rotation(image=image)
    writerLines = segment(image)

    num_lines_per_class += len(writerLines)

    boundingRects = np.asarray([])

    #
    # for line in writerLines:
    #     all_features_class = np.append(all_features_class, feature_extraction(line, image.shape))
    #     labels.append(class_num)
    #     num_training_examples += 1

    return np.reshape(all_features_class, (num_lines_per_class, num_features))


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
    normalized_X = np.divide(normalized_X, deviation)
    return normalized_X, mean, deviation


def reading_test_cases():
    global all_features
    global temp
    global num_training_examples
    global labels
    global num_lines_per_class
    global all_features_class
    results_array = []
    time_array = []

    indices_array = ['01', '02', '03', '04', '05', '06', '07', '08', '09']
    for i in range(10, 101):
        indices_array.append(str(i))

    for index in indices_array:
        # try:
            seconds = time.time()
            test_combination = (1, 2, 3)
            for class_number in test_combination:
                num_lines_per_class = 0
                all_features_class = np.asarray([])
                for filename in glob.glob('data/' + index + '/' + str(class_number) + '/*.jpg'):
                    print(filename)
                    temp = training(cv2.imread(filename), class_number)
                all_features = np.append(all_features,
                                         np.reshape(adjustNaNValues(temp), (1, num_lines_per_class * num_features)))

            # Normalization of features
            all_features, mu, sigma = featureNormalize(np.reshape(all_features, (num_training_examples, num_features)))

            classifier.fit(all_features, labels)
            labels = []
            all_features = []
            num_training_examples = 0
            total_cases = 0
            total_correct = 0
            for filename in glob.glob('data/' + index + '/test*.jpg'):
                print(filename)
                label = int(filename[len(filename) - 5])
                prediction = test(cv2.imread(filename), classifier, mu, sigma)
                print("label: " + str(label) + " prediction:" + str(prediction[0]))
                total_cases += 1
                if prediction[0] == label:
                    total_correct += 1
                results_array.append(str(prediction[0]) + '\n')
                print("Accuracy = ", total_correct * 100 / total_cases, " %")
            calculated_time = round(time.time() - seconds, 2)
            print("-----------------------------------------------------------------")
            print("Time:")
            print(calculated_time)
            time_array.append(str(calculated_time) + '\n')
            print("-----------------------------------------------------------------")
        # except:
        #     results_array.append(str(random.randint(1, 3)) + '\n')
        #     time_array.append(str(0) + '\n')

    time_file = open("time.txt", "w+")
    results_file = open("results.txt", "w+")
    time_file.writelines(time_array)
    time_file.close()
    results_file.writelines(results_array)
    results_file.close()


# Global Variables
all_features = np.asarray([])
all_features_class = np.asarray([])
labels = []
temp = []
num_training_examples = 0
num_features = 18
num_lines_per_class = 0
total_test_cases = 100

classifier = MLPClassifier(solver='lbfgs', max_iter=30000, alpha=0.046041, hidden_layer_sizes=(22,), random_state=1)
reading_test_cases()
