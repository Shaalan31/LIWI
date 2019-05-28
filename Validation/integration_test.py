from horestmodel.horest_model import *
from siftmodel.sift_model import *
from texturemodel.texture_model import *
from siftmodel.feature_matching import *

base_horest_training = "D:/Uni/Graduation Project/All Test Cases/Dataset/Validation/Samples/Horest/Class"
base_horest_testing = "D:/Uni/Graduation Project/All Test Cases/Dataset/Validation/TestCases/Horest/testing"
base_texture_training = "D:/Uni/Graduation Project/csv/Validation/Samples/H/0.5/Class"
base_texture_testing = "D:/Uni/Graduation Project/csv/Validation/backup/tstcases/0.5/"
base_SDS_training = 'D:/Uni/Graduation Project/csv/Validation/Samples/SDS/001/Class'
base_SDS_testing = 'D:/Uni/Graduation Project/csv/Validation/TestCases/SDS/001/'
base_SOH_training = 'D:/Uni/Graduation Project/csv/Validation/Samples/SOH/036/Class'
base_SOH_testing = 'D:/Uni/Graduation Project/csv/Validation/TestCases/SOH/036/'
matching = FeatureMatching()
accuracy = None

for radius in range(20, 170, 10):
    class_labels = list(range(1, 277))
    classCombinations = combinations(class_labels, r=radius)
    testcases = 0
    right_test_cases = 0
    total_test_cases = 0

    for class_combination in classCombinations:
        print(testcases)
        horest_model = HorestWriterIdentification()
        texture_model = TextureWriterIdentification()
        sift_model = SiftModel()

        SDS_train = []
        SOH_train = []
        for class_number in class_combination:
            horest_model.num_lines_per_class = 0
            texture_model.num_blocks_per_class = 0
            for filename in glob.glob(
                    base_horest_training + str(class_number) + '/*'):
                print(filename)
                name = Path(filename).name

                # horest
                horerst_features = np.genfromtxt(filename, delimiter=",")
                horest_model.num_lines_per_class += horerst_features.shape[0]
                for _ in range(horest_model.num_lines_per_class):
                    horest_model.labels.append(class_number)
                horest_model.all_features = np.append(horest_model.all_features,
                                                      np.reshape(horerst_features, (
                                                          1,
                                                          horest_model.num_lines_per_class * horest_model.num_features)))
                horest_model.num_training_examples += horest_model.num_lines_per_class

                # texture
                texture_features = np.genfromtxt(base_texture_training + str(class_number) + '/' + name, delimiter=",")
                texture_model.num_blocks_per_class += texture_features.shape[0]
                for _ in range(texture_model.num_blocks_per_class):
                    texture_model.labels.append(class_number)
                texture_model.all_features = np.append(texture_model.all_features,
                                                       np.reshape(texture_features, (
                                                           1,
                                                           texture_model.num_blocks_per_class * texture_model.num_features)))
                texture_model.num_training_examples += texture_model.num_blocks_per_class

                # sift
                name = name.replace(".csv", ".jpg")
                SDS_class = np.genfromtxt(base_SDS_training + str(class_number) + '/' + name, delimiter=",")
                SDS_class = SDS_class.reshape((1, SDS_class.shape[0]))
                SOH_class = np.genfromtxt(base_SOH_training + str(class_number) + '/' + name, delimiter=",")
                SOH_class = SOH_class.reshape((1, SOH_class.shape[0]))

                SDS_train.append(SDS_class)
                SOH_train.append(SOH_class)

        # Normalization of horest features
        horest_model.all_features, mu, sigma = horest_model.feature_normalize(
            np.reshape(horest_model.all_features, (horest_model.num_training_examples, horest_model.num_features)))

        horest_model.classifier.fit(horest_model.all_features, horest_model.labels)

        # Normalization of texture features
        texture_model.all_features, mu_texture, sigma_texture = texture_model.feature_normalize(
            np.reshape(texture_model.all_features, (texture_model.num_training_examples, texture_model.num_features)))
        texture_model.classifier.fit(texture_model.all_features, texture_model.labels)

        for class_number in class_combination:
            for filename in glob.glob(
                    base_horest_testing + str(
                        class_number) + '_*'):
                print(filename)
                name = Path(filename).name

                label = class_number

                # horest
                horest_features_test = np.genfromtxt(filename, delimiter=",")
                horest_features_test = (horest_features_test - mu) / sigma
                prediction_horest = np.average(horest_model.classifier.predict_proba(horest_features_test),
                                               axis=0).reshape(1, -1)
                classes_horest = horest_model.get_classifier_classes()
                prediction_horest = classes_horest[np.argmax(prediction_horest)]

                # texture
                texture_features_test = np.genfromtxt(base_texture_testing + name, delimiter=",")
                texture_features_test = (texture_features_test - mu_texture) / sigma_texture
                prediction_texture = np.average(texture_model.classifier.predict_proba(texture_features_test),
                                                axis=0).reshape(1, -1)
                classes_texture = texture_model.get_classifier_classes()
                # texture_indecies_sorted = np.argsort(classes_texture, axis=0)
                # sorted_texture_classes = classes_texture[texture_indecies_sorted[::-1]]
                # sorted_texture_predictions = prediction_texture[texture_indecies_sorted[::-1]]

                score = 0.2 * prediction_texture

                # sift
                # name = name.replace(".csv", ".jpg")
                SDS = np.genfromtxt(base_SDS_testing + name, delimiter=",")
                SDS = SDS.reshape((1, SDS.shape[0]))
                SOH = np.genfromtxt(base_SOH_testing + name, delimiter=",")
                SOH = SOH.reshape((1, SOH.shape[0]))

                # Feature Matching and Fusion
                manhattan = []
                chi_square = []
                for i in range(0, len(SDS_train)):
                    D1, D2 = matching.calculate_distances(u=SDS, v=SDS_train[i], x=SOH, y=SOH_train[i])
                    manhattan.append(D1)
                    chi_square.append(D2)
                prediction = matching.match(manhattan, chi_square, w=0.75)
                prediction_sift = class_combination[math.floor(prediction / 1)]

                score[0][np.argwhere(classes_texture == prediction_sift)[0]] += (0.4)
                score[0][np.argwhere(classes_texture == prediction_horest)[0]] += (0.4)
                final_prediction = int(classes_texture[np.argmax(score)])

                horest_model.total_test_cases += 1
                print(final_prediction)
                if final_prediction == label:
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
            np.savetxt('integration_result.csv', accuracy, delimiter=',')
            break
