from horestmodel.horest_model import *
import pickle
from siftmodel.sift_model_test import *



t = [1, 50, 150, 225, 300]
phi = [36, 72, 108, 234, 360]
w = [0.1,0.25,0.5,0.75,0.9]





code_book = pickle.load( open( "centers.pkl", "rb" ) )

#writer identification using SIFT

print('start')
class_labels = list(range(1, 301))
classCombinations = combinations(class_labels, r=300)
# total = len(classCombinations)
# print(total)
count = 0
for x in classCombinations:
    for t_test in t:
        accuracy_10 = None
        for phi_test in phi:
            accuracy = None
            count=0
            for w_test in w:
                print('t ',t_test,' - phi ',phi_test,' - w ',w_test)
                sift_model = SiftModel(test_classes=x , code_book=code_book,t=t_test,phi=phi_test,w=w_test)
                sift_model.run()
                if accuracy is None:
                    accuracy = sift_model.accuracy
                else:
                    accuracy = np.append(accuracy,sift_model.accuracy,axis=1)
                print('Total accuracy',accuracy[0,count]/accuracy[1,count])
                np.savetxt("accuracyTester.csv", accuracy[0,:]/accuracy[1,:], delimiter=",")
                #np.savetxt("varinacc.csv", sift_model.thesis, delimiter=",")
                count +=1
                print(accuracy.shape)
            accuracy = accuracy[0, :] / accuracy[1, :]
            if accuracy_10 is None:
                accuracy_10 = accuracy
            else:
                accuracy_10 = np.append(accuracy_10, accuracy, axis=0)
            print('accuracy_10 shape ',accuracy_10.shape)
            np.savetxt("t"+str(t_test)+'p'+str(phi_test)+'.csv',accuracy_10,delimiter=",")

#Validate Texture


# texture_model = TextureWriterIdentification('D:/Uni/Graduation Project/All Test Cases/KHATT/Samples/Class',
#                                             # 'D:/Uni/Graduation Project/All Test Cases/KHATT/TestCases/testing')
#                                             'C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/KHATT/TestCases/testing')
# textureMethod.run()


# SAMAR
# code book
#code_book = pickle.load(open("siftmodel/centers.pkl", "rb"))


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

#
# def preprocess_texture(h,list_classes):
#     print("Preprocessing")
#     base_samples_h = 'C:/Users/omars/Documents/Github/LIWI/Omar/Validation/Samples/H/'
#
#     h_coeff = int(h)
#     # declaring variables for texture
#     labels_texture = []
#     all_features_texture = []
#     num_training_examples_texture = 0
#
#     for writer_id in list_classes:
#         for filename in glob.glob(base_samples_h+str(h_coeff)+"/Class"+str(writer_id)+'/'+name):
#             print(filename)
#             # texture_features = writer.features.texture_feature
#             texture_features = np.genfromtxt(filename)
#             num_current_examples_texture = len(texture_features)
#             labels_texture = np.append(labels_texture,
#                                        np.full(shape=(1, num_current_examples_texture), fill_value=writer_id))
#             num_training_examples_texture += num_current_examples_texture
#             all_features_texture = np.append(all_features_texture,
#                                              np.reshape(texture_features.copy(),
#                                                         (1,
#                                                          num_current_examples_texture * texture_model.get_num_features())))
#
#
#     # fit texture classifier
#     all_features_texture = np.reshape(all_features_texture,
#                                       (num_training_examples_texture, texture_model.get_num_features()))
#     all_features_texture, mu_texture, sigma_texture = texture_model.feature_normalize(all_features_texture)
#     pca = decomposition.PCA(n_components=min(all_features_texture.shape[0], all_features_texture.shape[1]),
#                             svd_solver='full')
#     all_features_texture = pca.fit_transform(all_features_texture)
#     texture_model.fit_classifier(all_features_texture, labels_texture)
#
#     return mu_texture, sigma_texture, pca, writers_lookup_array
#
#
# def predict_texture(testing_image, mu_texture, sigma_texture, pca):
#
#
#     print("Starting Texture Testing")
#     texture_classes = texture_model.get_classifier_classes()
#     texture_predictions = texture_model.test(testing_image, mu_texture, sigma_texture,pca)[0]
#     texture_indecies_sorted = np.argsort(texture_classes, axis=0)
#     sorted_texture_predictions = texture_predictions[texture_indecies_sorted[::-1]]
#     sorted_texture_classes = texture_classes[texture_indecies_sorted[::-1]]
#     print("Texture Prediction:" + str(sorted_texture_classes[np.argmax(sorted_texture_predictions)]))
#
#     # score = 0.25 * sorted_horest_predictions + 0.25 * sorted_texture_predictions
#
#     # print("Starting Sift Testing")
#     # sift_prediction = sift_model.predict(SDS_train, SOH_train, testing_image, filename,lang="en")
#     # sift_prediction = writers_lookup_array[sift_prediction]
#     # print("Sift Prediction:" + str(sift_prediction))
#
#     # score[np.argwhere(sorted_texture_classes == sift_prediction)] += (1 / 2)
#     # final_prediction = int(sorted_horest_classes[np.argmax(score)])
#     final_prediction = sorted_texture_classes[np.argmax(sorted_texture_predictions)]
#     print("Common Prediction: " + str(final_prediction))
#
#     return final_prediction
#


# English

# class_labels = list(range(1, 300))
# classCombinations = combinations(class_labels, r=300)
#
#
# h = [0.1, 0.3, 0.5, 0.7, 0.9]
# for h_coeff in h:
#     mu_texture, sigma_texture, pca, writers_lookup_array = preprocess_texture(h_coeff,classCombinations)
#
#     for count in classCombinations:
#         print('Class' + str(count) + ':')
#
#         for filename in glob.glob(sift_model.base_test+ 'testing' + str(count) + '_*.jpg'):
#             name = Path(filename).name
#             print(name)
#             image = cv2.imread(filename)
#             prediction = predict_texture(image, name, mu_horest, sigma_horest, mu_texture, sigma_texture, pca, SDS_train, SOH_train, writers_lookup_array)
#
#             if (prediction == count):
#                 right_test_cases += 1
#             total_test_cases += 1
#
#             accuracy = (right_test_cases / total_test_cases) * 100
#
#             print("Accuracy: " + str(accuracy) + "%")
#
#         count += 1
# End English



#Validate Sift



#
# def validate_sift(t,phi,w,writers_list):
#     print("Sift")
#
#     base_samples_t = 'C:/Users/omars/Documents/Github/LIWI/Omar/Validation/Samples/SDS/'
#     base_samples_phi = 'C:/Users/omars/Documents/Github/LIWI/Omar/Validation/Samples/SOH/'
#
#     # declaring variable for sift
#     SDS_train = []
#     SOH_train = []
#     writers_lookup_array = []
#
#     t = [1, 50, 150, 225, 300]
#     phi = [36, 72, 108, 234, 360]
#     w = [0.1,0.25,0.5,0.75,0.9]
#     # set english code book for sift
#     sift_model.set_code_book("en")
#
#     for test_t in t:
#         str_t = str(test_t)
#         while len(str_t) < 3:
#             str_t = '0' + str_t
#         #read another SDS
#         for writer_id in writers_list:
#             for filename in glob.glob(base_samples_t + str_t + "/Class" + str(writer_id) + '/*jpg'):
#
#
#
#
#
#         for test_phi in phi:
#             #read another SOH
#             for test_w in w:
#                 #match
#
# def preprocess_sift(t,phi,w,writers_list):
#
#
#     str_phi = str(phi[idx])
#
#
#     while len(str_phi) < 3:
#         str_phi = '0' + str_phi
#
#     base_samples_t + str_t + "/Class" + str(class_number) + '/' + name
#     base_samples_phi + str_phi + "/Class" + str(class_number) + '/' + name
#
#     for writer_id in writers_list:
#         for filename in glob.glob(base_samples_h + str(h_coeff) + "/Class" + str(writer_id) + '/' + name):
#             # processing texture_features
#             texture_features = writer.features.texture_feature
#             num_current_examples_texture = len(texture_features)
#             labels_texture = np.append(labels_texture,
#                                        np.full(shape=(1, num_current_examples_texture), fill_value=writer.id))
#             num_training_examples_texture += num_current_examples_texture
#             all_features_texture = np.append(all_features_texture,
#                                              np.reshape(texture_features.copy(),
#                                                         (1,
#                                                          num_current_examples_texture * texture_model.get_num_features())))
#
#             # appending sift features
#             for i in range(len(writer.features.sift_SDS)):
#                 SDS_train.append(np.array([writer.features.sift_SDS[i]]))
#                 SOH_train.append(np.array([writer.features.sift_SOH[i]]))
#                 writers_lookup_array.append(writer.id)
#
#     return SDS_train, SOH_train, writers_lookup_array
#
#
# def predict_sift(testing_image, filename, mu_horest, sigma_horest, mu_texture, sigma_texture, pca, SDS_train, SOH_train, writers_lookup_array):
#
#     # used to match the probablity with classes
#     # print("Starting Horest Testing")
#     # horest_classes = horest_model.get_classifier_classes()
#     # horest_predictions = horest_model.test(testing_image, mu_horest, sigma_horest)[0]
#     # horest_indecies_sorted = np.argsort(horest_classes, axis=0)
#     # sorted_horest_classes = horest_classes[horest_indecies_sorted[::-1]]
#     # sorted_horest_predictions = horest_predictions[horest_indecies_sorted[::-1]]
#     # print("Horest Prediction: " + str(sorted_horest_classes[np.argmax(sorted_horest_predictions)]))
#
#     # print("Starting Texture Testing")
#     texture_classes = texture_model.get_classifier_classes()
#     texture_predictions = texture_model.test(testing_image, mu_texture, sigma_texture,pca)[0]
#     texture_indecies_sorted = np.argsort(texture_classes, axis=0)
#     sorted_texture_predictions = texture_predictions[texture_indecies_sorted[::-1]]
#     sorted_texture_classes = texture_classes[texture_indecies_sorted[::-1]]
#     print("Texture Prediction:" + str(sorted_texture_classes[np.argmax(sorted_texture_predictions)]))
#
#     # score = 0.25 * sorted_horest_predictions + 0.25 * sorted_texture_predictions
#
#     # print("Starting Sift Testing")
#     # sift_prediction = sift_model.predict(SDS_train, SOH_train, testing_image, filename,lang="en")
#     # sift_prediction = writers_lookup_array[sift_prediction]
#     # print("Sift Prediction:" + str(sift_prediction))
#
#     # score[np.argwhere(sorted_texture_classes == sift_prediction)] += (1 / 2)
#     # final_prediction = int(sorted_horest_classes[np.argmax(score)])
#     final_prediction = sorted_texture_classes[np.argmax(sorted_texture_predictions)]
#     print("Common Prediction: " + str(final_prediction))
#
#     return final_prediction
#
#
