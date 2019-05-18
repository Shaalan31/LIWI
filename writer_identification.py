from texturemodel.texture_model import *
from horestmodel.horest_model import *
from multiprocessing import Pool
from siftmodel.sift_model import *
import pickle
import cv2
from server.dao.connection import Database
from server.dao.writers import Writers

texture_model = TextureWriterIdentification('D:/Uni/Graduation Project/All Test Cases/KHATT/Samples/Class',
                                            # 'D:/Uni/Graduation Project/All Test Cases/KHATT/TestCases/testing')
                                            'D:/Uni/Graduation Project/All Test Cases/KHATT/TestCases/testing')
# textureMethod.run()

# Horst Writer identification
horest_model = HorestWriterIdentification('D:/Uni/Graduation Project/All Test Cases/IAM/Samples/Class',
                                          'D:/Uni/Graduation Project/All Test Cases/IAM/TestCases/testing')
# horestMethod.run()

# SAMAR
# code book
code_book = pickle.load(open("siftmodel/centers.pkl", "rb"))
sift_model = SiftModel(first_class=91, last_class=121)


# sift_model.run()

# sift_model.get_features("C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/Samples/Class1/a01-000u.png","a01-000u.png")
# sift_model.get_features("C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/Samples/Class75/f04-011.png","f04-011.png")
# sift_model.get_features("C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/Samples/Class114/g07-022b.png","g07-022b.png")
# sift_model.get_features("C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/Samples/Class112/g07-003b.png","g07-003b.png")
# sift_model.get_features("C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/TestCases/testing24_2.png","testing24_2.png")
# sift_model.get_features("C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/TestCases/testing48_3.png","testing48_3.png")
# sift_model.get_features("C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/TestCases/testing1_27.png","testing1_27.png")
# sift_model.get_features("C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/TestCases/testing1_9.png","testing1_9.png")
# sift_model.get_features("C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/KHATT/TestCases/testing1.png","testing1.png")
# sift_model.get_features("C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/KHATT/TestCases/testing290.png","testing290.png")
# sift_model.get_features("C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/KHATT/TestCases/testing798.png","testing798.png")
# sift_model.get_features("C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/KHATT/TestCases/testing187.png","testing187.png")

# End Samar

# Shaalan
# code_book = pickle.load( open( "siftmodel/centers.pkl", "rb" ) )
#
# #writer identification using SIFT
# accuracy = None
# for x in range(1,160,10):
#     sift_model = SiftModel(first_class=x , last_class=x+9, code_book=code_book)
#     sift_model.run()
#     if accuracy is None:
#         accuracy = sift_model.accuracy
#     else:
#         accuracy = np.append(accuracy,sift_model.accuracy,axis=1)
#     print('Total accuracy',accuracy)
#     print(accuracy.shape)
#
# np.savetxt("accuracy6.csv", accuracy, delimiter=",")


def predict_writer(testing_image, filename, writers):

    # process features to fit classifier
    # declaring variables for horest
    labels_horest = []
    all_features_horest = []
    num_training_examples_horest = 0

    # declaring variables for texture
    labels_texture = []
    all_features_texture = []
    num_training_examples_texture = 0

    # declaring variable for sift
    SDS_train = []
    SOH_train = []
    writers_lookup_array = []
    # set english code book for sift
    sift_model.set_code_book("en")

    for writer in writers:
        # processing horest_features
        horest_features = writer.features.horest_features
        num_current_examples_horest = len(horest_features)
        labels_horest = np.append(labels_horest,
                                  np.full(shape=(1, num_current_examples_horest), fill_value=writer.id))
        num_training_examples_horest += num_current_examples_horest
        all_features_horest = np.append(all_features_horest,
                                        np.reshape(horest_features.copy(),
                                                   (1, num_current_examples_horest * horest_model.get_num_features())))
        # processing texture_features
        texture_features = writer.features.texture_feature
        num_current_examples_texture = len(texture_features)
        labels_texture = np.append(labels_texture,
                                   np.full(shape=(1, num_current_examples_texture), fill_value=writer.id))
        num_training_examples_texture += num_current_examples_texture
        all_features_texture = np.append(all_features_texture,
                                         np.reshape(texture_features.copy(),
                                                    (1,
                                                     num_current_examples_texture * texture_model.get_num_features())))

        # appending sift features
        for i in range(len(writer.features.sift_SDS)):
            SDS_train.append(np.array([writer.features.sift_SDS[i]]))
            SOH_train.append(np.array([writer.features.sift_SOH[i]]))
            writers_lookup_array.append(writer.id)

    # fit horest classifier
    all_features_horest = np.reshape(all_features_horest,
                                     (num_training_examples_horest, horest_model.get_num_features()))
    all_features_horest, mu_horest, sigma_horest = horest_model.feature_normalize(all_features_horest)
    horest_model.fit_classifier(all_features_horest, labels_horest)

    # fit texture classifier
    all_features_texture = np.reshape(all_features_texture,
                                      (num_training_examples_texture, texture_model.get_num_features()))
    all_features_texture, mu_texture, sigma_texture = texture_model.feature_normalize(all_features_texture)
    pca = decomposition.PCA(n_components=min(all_features_texture.shape[0], all_features_texture.shape[1]),
                            svd_solver='full')
    all_features_texture = pca.fit_transform(all_features_texture)
    texture_model.fit_classifier(all_features_texture, labels_texture)

    # used to match the probablity with classes
    print("Starting Horest Testing")
    horest_classes = horest_model.get_classifier_classes()
    horest_predictions = horest_model.test(testing_image, mu_horest, sigma_horest)[0]
    horest_indecies_sorted = np.argsort(horest_classes, axis=0)
    sorted_horest_classes = horest_classes[horest_indecies_sorted[::-1]]
    sorted_horest_predictions = horest_predictions[horest_indecies_sorted[::-1]]
    print("Horest Prediction: " + str(sorted_horest_classes[np.argmax(sorted_horest_predictions)]))

    print("Starting Texture Testing")
    texture_classes = texture_model.get_classifier_classes()
    texture_predictions = texture_model.test(testing_image, mu_texture, sigma_texture, pca)[0]
    texture_indecies_sorted = np.argsort(texture_classes, axis=0)
    sorted_texture_predictions = texture_predictions[texture_indecies_sorted[::-1]]
    sorted_texture_classes = texture_classes[texture_indecies_sorted[::-1]]
    print("Texture Prediction:" + str(sorted_texture_classes[np.argmax(sorted_texture_predictions)]))

    score = 0.25 * sorted_horest_predictions + 0.25 * sorted_texture_predictions

    print("Starting Sift Testing")
    sift_prediction = sift_model.predict(SDS_train, SOH_train, testing_image, filename,lang="en")
    sift_prediction = writers_lookup_array[sift_prediction]
    print("Sift Prediction:" + str(sift_prediction))

    score[np.argwhere(sorted_texture_classes == sift_prediction)] += (1 / 2)
    final_prediction = int(sorted_horest_classes[np.argmax(score)])
    print("Common Prediction: " + str(final_prediction))

    return final_prediction


def predict_writer_arabic(testing_image, filename, writers_ids, dao):
    writers = dao.get_features(writers_ids)

    # process features to fit classifier
    # declaring variables for texture
    labels_texture = []
    all_features_texture = []
    num_training_examples_texture = 0

    # declaring variable for sift
    SDS_train = []
    SOH_train = []
    writers_lookup_array = []
    # set arabic code book for sift
    sift_model.set_code_book("ar")

    for writer in writers:
         # processing texture_features
        texture_features = writer.features.texture_feature
        num_current_examples_texture = len(texture_features)
        labels_texture = np.append(labels_texture,
                                   np.full(shape=(1, num_current_examples_texture), fill_value=writer.id))
        num_training_examples_texture += num_current_examples_texture
        all_features_texture = np.append(all_features_texture,
                                         np.reshape(texture_features.copy(),
                                                    (1,
                                                     num_current_examples_texture * texture_model.get_num_features())))

        # appending sift features
        for i in range(len(writer.features.sift_SDS)):
            SDS_train.append(np.array([writer.features.sift_SDS[i]]))
            SOH_train.append(np.array([writer.features.sift_SOH[i]]))
            writers_lookup_array.append(writer.id)

    # fit texture classifier
    all_features_texture = np.reshape(all_features_texture,
                                      (num_training_examples_texture, texture_model.get_num_features()))
    all_features_texture, mu_texture, sigma_texture = texture_model.feature_normalize(all_features_texture)
    pca = decomposition.PCA(n_components=min(all_features_texture.shape[0], all_features_texture.shape[1]),
                            svd_solver='full')
    all_features_texture = pca.fit_transform(all_features_texture)
    texture_model.fit_classifier(all_features_texture, labels_texture)


    print("Starting Texture Testing")
    texture_classes = texture_model.get_classifier_classes()
    texture_predictions = texture_model.test(testing_image, mu_texture, sigma_texture, pca,lang="ar")[0]
    texture_indecies_sorted = np.argsort(texture_classes, axis=0)
    sorted_texture_predictions = texture_predictions[texture_indecies_sorted[::-1]]
    sorted_texture_classes = texture_classes[texture_indecies_sorted[::-1]]
    print("Texture Prediction:" + str(sorted_texture_classes[np.argmax(sorted_texture_predictions)]))

    score =  0.6 * sorted_texture_predictions

    print("Starting Sift Testing")
    sift_prediction = sift_model.predict(SDS_train, SOH_train, testing_image, filename,lang="ar")
    sift_prediction = writers_lookup_array[sift_prediction]
    print("Sift Prediction:" + str(sift_prediction))

    score[np.argwhere(sorted_texture_classes == sift_prediction)] += 0.4
    final_prediction = int(sorted_texture_classes[np.argmax(score)])
    print("Common Prediction: " + str(final_prediction))

    return final_prediction

def predict_writer_arabic_edit(testing_image, filename, writers, correct_sift_cases, correct_texture_cases, label):

    # process features to fit classifier
    # declaring variables for texture
    labels_texture = []
    all_features_texture = []
    num_training_examples_texture = 0

    # declaring variable for sift
    SDS_train = []
    SOH_train = []
    writers_lookup_array = []
    # set arabic code book for sift
    sift_model.set_code_book("ar")

    for writer in writers:
         # processing texture_features
        texture_features = writer.features.texture_feature
        num_current_examples_texture = len(texture_features)
        labels_texture = np.append(labels_texture,
                                   np.full(shape=(1, num_current_examples_texture), fill_value=writer.id))
        num_training_examples_texture += num_current_examples_texture
        all_features_texture = np.append(all_features_texture,
                                         np.reshape(texture_features.copy(),
                                                    (1,
                                                     num_current_examples_texture * texture_model.get_num_features())))

        # appending sift features
        for i in range(len(writer.features.sift_SDS)):
            SDS_train.append(np.array([writer.features.sift_SDS[i]]))
            SOH_train.append(np.array([writer.features.sift_SOH[i]]))
            writers_lookup_array.append(writer.id)

    # fit texture classifier
    all_features_texture = np.reshape(all_features_texture,
                                      (num_training_examples_texture, texture_model.get_num_features()))
    all_features_texture, mu_texture, sigma_texture = texture_model.feature_normalize(all_features_texture)
    pca = decomposition.PCA(n_components=min(all_features_texture.shape[0], all_features_texture.shape[1]),
                            svd_solver='full')
    all_features_texture = pca.fit_transform(all_features_texture)
    texture_model.fit_classifier(all_features_texture, labels_texture)


    print("Starting Texture Testing")
    texture_classes = texture_model.get_classifier_classes()
    texture_predictions = texture_model.test(testing_image, mu_texture, sigma_texture, pca,lang="ar")[0]
    texture_indecies_sorted = np.argsort(texture_classes, axis=0)
    sorted_texture_predictions = texture_predictions[texture_indecies_sorted[::-1]]
    sorted_texture_classes = texture_classes[texture_indecies_sorted[::-1]]
    print("Texture Prediction:" + str(sorted_texture_classes[np.argmax(sorted_texture_predictions)]))
    if(sorted_texture_classes[np.argmax(sorted_texture_predictions)] == label):
        correct_texture_cases += 1

    score =  0.6 * sorted_texture_predictions

    print("Starting Sift Testing")
    sift_prediction = sift_model.predict(SDS_train, SOH_train, testing_image, filename,lang="ar")
    sift_prediction = writers_lookup_array[sift_prediction]
    print("Sift Prediction:" + str(sift_prediction))
    if(sift_prediction == label):
        correct_sift_cases += 1

    score[np.argwhere(sorted_texture_classes == sift_prediction)] += 0.4
    final_prediction = int(sorted_texture_classes[np.argmax(score)])
    print("Common Prediction: " + str(final_prediction))

    return final_prediction, correct_sift_cases, correct_texture_cases


# db = Database()
# db.connect()
# db.create_collection()
# writers_dao = Writers(db.get_collection())
# first_class = 2
# last_class = 100
# count = first_class
# total_test_cases = 0
# right_test_cases = 0
#
# writers = writers_dao.get_features(list(range(first_class, last_class + 1)))
#
# while count <= last_class:
#     print('Class' + str(count) + ':')
#
#     for filename in glob.glob(sift_model.base_test+ 'testing' + str(count) + '_*.jpg'):
#         name = Path(filename).name
#         print(name)
#         image = cv2.imread(filename)
#         prediction = predict_writer(image, name, writers)
#
#         if (prediction == count):
#             right_test_cases += 1
#         total_test_cases += 1
#
#         accuracy = (right_test_cases / total_test_cases) * 100
#
#         print("Accuracy: " + str(accuracy) + "%")
#
#     count += 1

# Samar
# db = Database()
# db.connect()
# db.create_collection()
# writers_dao = Writers(db.get_collection_arabic())
# startClass = 2
# endClass = 100
# count = startClass
#
# total_cases = 0
# total_correct = 0
# correct_sift_cases = 0
# correct_texture_cases = 0
#
# writers = writers_dao.get_features(list(range(startClass, endClass + 1)))
#
# while count <= endClass:
#
#     print('Class' + str(count) + ':')
#     for filename in glob.glob(
#             texture_model.pathTestCases + str(
#                 count) + '.png'):
#         print(filename)
#         label = count
#         prediction, correct_sift_cases, correct_texture_cases = predict_writer_arabic_edit(cv2.imread(filename),filename, writers, correct_sift_cases, correct_texture_cases, label)
#
#         total_cases += 1
#         if prediction == label:
#             total_correct += 1
#         print("Accuracy = ", total_correct * 100 / total_cases, " %")
#
#         print("Accuracy Sift: ", correct_sift_cases * 100 / total_cases, "%")
#         print("Accuracy Texture: ", correct_texture_cases * 100 / total_cases, "%")
#
#     count += 1
# End Samar

# May
db = Database()
db.connect()
db.create_collection()
writers_dao = Writers(db.get_collection_arabic())
startClass = 1
endClass = 20
classCombinations = combinations(range(startClass, endClass + 1), r=10)

total_cases = 0
total_correct = 0

for classCombination in classCombinations:
    for class_number in classCombination:
        for filename in glob.glob(
                texture_model.pathTestCases + str(
                    class_number) + '.png'):
            print(filename)
            label = class_number
            prediction = predict_writer_arabic(cv2.imread(filename),filename, list(classCombination),writers_dao)
            total_cases += 1
            if prediction == label:
                total_correct += 1
            print("Accuracy = ", total_correct * 100 / total_cases, " %")
# End May