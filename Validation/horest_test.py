from horestmodel.horest_model import *

accuracy = None
for radius in range(10, 170, 10):
    class_labels = list(range(1, 290))
    classCombinations = combinations(class_labels, r=radius)
    testcases = 0
    right_test_cases = 0
    total_test_cases = 0

    for class_combination in classCombinations:
        print(testcases)
        horest_model = HorestWriterIdentification(
            path_training_set="D:/Uni/Graduation Project/All Test Cases/Dataset/Validation/Samples/Horest/Class",
            path_test_cases="D:/Uni/Graduation Project/All Test Cases/Dataset/Validation/TestCases/Horest/testing")

        for class_number in class_combination:
            horest_model.num_lines_per_class = 0
            for filename in glob.glob(
                    horest_model.pathTrainingSet + str(class_number) + '/*'):
                print(filename)
                features = np.genfromtxt(filename, delimiter=",")
                horest_model.num_lines_per_class += features.shape[0]
                for i in range(horest_model.num_lines_per_class):
                    horest_model.labels.append(class_number)
                horest_model.all_features = np.append(horest_model.all_features,
                                                      np.reshape(features,
                                                                 (1, horest_model.num_lines_per_class * horest_model.num_features)))
                horest_model.num_training_examples += horest_model.num_lines_per_class
        # Normalization of features
        horest_model.all_features, mu, sigma = horest_model.feature_normalize(
            np.reshape(horest_model.all_features, (horest_model.num_training_examples, horest_model.num_features)))

        horest_model.classifier.fit(horest_model.all_features, horest_model.labels)

        for class_number in class_combination:
            for filename in glob.glob(
                    horest_model.pathTestCases + str(
                        class_number) + '_*'):
                print(filename)
                label = class_number
                all_features_test = np.genfromtxt(filename, delimiter=",")
                all_features_test = (all_features_test - mu) / sigma
                prediction = np.average(horest_model.classifier.predict_proba(all_features_test), axis=0).reshape(1, -1)
                classes = horest_model.get_classifier_classes()
                prediction = classes[np.argmax(prediction)]
                horest_model.total_test_cases += 1
                print(prediction)
                if prediction == label:
                    horest_model.right_test_cases += 1


        testcases += horest_model.total_test_cases
        right_test_cases += horest_model.right_test_cases
        if testcases > 5000:

            if accuracy is None:
                accuracy = np.array([radius, right_test_cases / testcases]).reshape((1, 2))
            else:
                accuracy = np.append(accuracy, np.array([radius, right_test_cases / testcases]).reshape((1, 2)),
                                     axis=0)
            print('Acc finaal @', ' - ', right_test_cases / testcases, 'rad - ', radius)
            print('shape ', accuracy.shape)
            np.savetxt('horest_nn_results.csv', accuracy, delimiter=',')
            break
