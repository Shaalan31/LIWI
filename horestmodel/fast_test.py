import warnings
from itertools import combinations
from sklearn.neural_network import MLPClassifier
from horestmodel.line_segmentation import *
from horestmodel.horest_features import *
import glob

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Global Variables
num_features = 18
num_lines_per_class = 0
num_classes = 50
training_dict = {}
testing_dict = {}
randomState = 1545481387


def process_training_data():
    global num_lines_per_class

    for class_number in range(1, num_classes + 1):
        temp = np.asarray([])
        num_lines_per_class = 0
        for filename in glob.glob(
                'D:/Uni/Graduation Project/All Test Cases/IAMJPG/Samples/Class' + str(class_number) + '/*.jpg'):
            print(filename)
            temp = np.append(temp, training(cv2.imread(filename)))
        training_dict[class_number] = adjustNaNValues(np.reshape(temp, (num_lines_per_class, num_features)))


def process_test_data():
    for class_number in range(1, num_classes + 1):
        temp = np.asarray([])
        for filename in glob.glob(
                'D:/Uni/Graduation Project/All Test Cases/IAMJPG/TestCases/testing' + str(class_number) + '_*.jpg'):
            temp = test(cv2.imread(filename))
            break
        testing_dict[class_number] = temp

def feature_extraction(example, image_shape):

    example = example.astype('uint8')
    example_copy = example.copy()

    # Calculate Contours
    contours, hierarchy = cv2.findContours(example_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]
    contours = np.asarray(contours)

    horest_features = HorestFeatures(example, contours, hierarchy)
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


def test(image):
    all_features_test = np.asarray([])

    if image.shape[0] > 3500:
        image = cv2.resize(src=image, dsize=(3500, round((3500 / image.shape[1]) * image.shape[0])))

    # image = adjust_rotation(image=image)
    # show_images([image])
    writer_lines = LineSegmentation(image).segment()

    num_testing_examples = 0
    for line in writer_lines:
        example = feature_extraction(line, image.shape)
        all_features_test = np.append(all_features_test, example)
        num_testing_examples += 1

    all_features_test = adjust_nan_values(np.reshape(all_features_test, (num_testing_examples, num_features)))

    return np.average(all_features_test, axis=0).reshape(1, -1)


def training(image):
    global num_lines_per_class

    image_height = image.shape[0]
    if image_height > 3500:
        image = cv2.resize(src=image, dsize=(3500, round((3500 / image.shape[1]) * image_height)))

    # image = adjust_rotation(image=image)
    # show_images([image])
    writer_lines = LineSegmentation(image).segment()
    num_lines = len(writer_lines)

    all_features_class = np.asarray([])
    num_lines_per_class += len(writer_lines)
    for line in writer_lines:
        all_features_class = np.append(all_features_class, feature_extraction(line, image.shape))


    return np.reshape(all_features_class, (1, num_lines*num_features))



def adjust_nan_values(writer_features):
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


def start():
    correct_answers = 0
    total_answers = 0
    class_labels = list(range(1, num_classes + 1))
    classCombinations = combinations(class_labels, r=30)
    classifier = MLPClassifier(solver='lbfgs', max_iter=30000, alpha=0.046041,
                               hidden_layer_sizes=(22, 18, 15, 12, 7,),
                               random_state=randomState)
    avg = np.array([])
    for class_combination in classCombinations:
        accuracy = 0

        all_features = np.asarray([])
        labels = np.asarray([])
        num_training_examples = 0

        # print(test_combination)
        for class_number in class_combination:
            num_current_examples = len(training_dict[class_number])
            labels = np.append(labels, np.full(shape=(1, num_current_examples), fill_value=class_number))
            num_training_examples += num_current_examples
            all_features = np.append(all_features,
                                     np.reshape(training_dict[class_number].copy(),
                                                (1, num_current_examples * num_features)))

        all_features = np.reshape(all_features, (num_training_examples, num_features))
        all_features, mu, sigma = featureNormalize(all_features)

        classifier.fit(all_features, labels)

        for class_number in class_combination:
            # print(test_combination[classNum])
            test_vector = (testing_dict[class_number]).copy()
            test_vector = (test_vector - mu) / sigma
            prediction = classifier.predict_proba(test_vector.reshape(1, -1))
            classes = classifier.classes_
            prediction=classes[np.argmax(prediction)]
            # print(prediction)

            if prediction == class_number:
                correct_answers += 1
            # else:
            #     file = open("wrngClassified.txt", "a")
            #     file.write(str(test_combination))
            #     file.write('\n')
            #     file.close()
            total_answers += 1
            accuracy = (correct_answers / total_answers) * 100
            print("Accuracy = ", accuracy, "%")
    # avg = np.append(avg, accuracy)
    correct_answers = 0
    total_answers = 0
    # print(i)
    # np.savetxt("fasttest.csv", avg, delimiter=",")


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

#
# process_training_data()
# for key, value in training_dict.items():
#     np.savetxt("D:/Uni/Graduation Project/All Test Cases/IAMCSV/training" + str(key) + ".csv", value, delimiter=",")

# process_test_data()
# for key, value in testing_dict.items():
#     np.savetxt("D:/Uni/Graduation Project/All Test Cases/IAMCSV/test" + str(key) + ".csv", value, delimiter=",")


for i in range(1, num_classes + 1):
    print(i)
    training_dict[i] = np.genfromtxt('D:/Uni/Graduation Project/All Test Cases/IAMCSV/Samples/Horest/training' + str(i) + '.csv', delimiter=",")
    testing_dict[i] = np.genfromtxt('D:/Uni/Graduation Project/All Test Cases/IAMCSV/TestCases/Horest/test' + str(i) + '.csv', delimiter=",")
start()
