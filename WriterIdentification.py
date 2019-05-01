from textureModel.TextureMethod import *

textureMethod = TextureWriterIdentification('D:/Uni/Graduation Project/All Test Cases/KHATT/Samples/Class','D:/Uni/Graduation Project/All Test Cases/KHATT/TestCases/testing')
textureMethod.run()

# import glob
# import warnings
# from itertools import combinations
# from sklearn import decomposition
# from sklearn import svm
# from lpq import *
# from segmentation import *
#
# warnings.filterwarnings("ignore", category=RuntimeWarning)
#
#
# def feature_extraction(example):
#     example = example.astype('uint8')
#     example_copy = example.copy()
#
#     feature = []
#
#     lbp = local_binary_pattern(example_copy, P=8, R=1)
#
#     feature.extend(np.histogram(lbp, bins=256)[0])
#
#     lpq = LPQ(radius=1).__call__(example_copy)
#     feature.extend(np.histogram(np.array(lpq), bins=256)[0])
#
#     return np.asarray(feature)
#
#
# def test(image, clf, mu, sigma, pca):
#     all_features_test = np.asarray([])
#
#     if image.shape[0] > 3500:
#         image = cv2.resize(src=image, dsize=(3500, round((3500 / image.shape[1]) * image.shape[0])))
#
#     # image = adjust_rotation(image=image)
#     writer_blocks = segment(image)
#
#     num_testing_examples = 0
#     for block in writer_blocks:
#         example = feature_extraction(block)
#         all_features_test = np.append(all_features_test, example)
#         num_testing_examples += 1
#
#     all_features_test = (adjustNaNValues(
#         np.reshape(all_features_test, (num_testing_examples, num_features))) - mu) / sigma
#
#     # Predict on each line
#     # predictions = []
#     # for example in all_features_test:
#     #     predictions.append(clf.predict(np.asarray(example).reshape(1, -1)))
#     # values, counts = np.unique(np.asarray(predictions), return_counts=True)
#     # return values[np.argmax(counts)]
#     return clf.predict(pca.transform(np.average(all_features_test, axis=0).reshape(1, -1)))
#
#
# def training(image, class_num):
#     global all_features
#     global all_features_class
#     global labels
#     global num_blocks_per_class
#     global num_training_examples
#
#     image_height = image.shape[0]
#     if image_height > 3500:
#         image = cv2.resize(src=image, dsize=(3500, round((3500 / image.shape[1]) * image_height)))
#
#     # image = adjust_rotation(image=image)
#     writer_blocks = segment(image)
#
#     num_blocks_per_class += len(writer_blocks)
#
#     for block in writer_blocks:
#         all_features_class = np.append(all_features_class, feature_extraction(block))
#         labels.append(class_num)
#         num_training_examples += 1
#
#     return np.reshape(all_features_class, (num_blocks_per_class, num_features))
#
#
# def adjustNaNValues(writer_features):
#     for i in range(num_features):
#         feature = writer_features[:, i]
#         is_nan_mask = np.isnan(feature)
#         if len(np.where(np.asarray(is_nan_mask))[0]) == 0:
#             continue
#
#         non_nan_indices = np.where(np.logical_not(is_nan_mask))[0]
#         nan_indices = np.where(is_nan_mask)[0]
#
#         if len(non_nan_indices) == 0:
#             feature_mean = 0
#         else:
#             feature_mean = np.mean(feature[non_nan_indices])
#         writer_features[nan_indices, i] = feature_mean
#
#     return writer_features
#
#
# def featureNormalize(X):
#     mean = np.mean(X, axis=0)
#     normalized_X = X - mean
#     deviation = np.sqrt(np.var(normalized_X, axis=0))
#     if len(np.where(deviation == 0)[0]) > 0:
#         deviation[np.where(deviation == 0)[0]] = 1
#     normalized_X = np.divide(normalized_X, deviation)
#     return normalized_X, mean, deviation
#
#
# randomState = 1545481387
#
#
# def reading_test_cases():
#     global all_features
#     global temp
#     global num_training_examples
#     global labels
#     global num_blocks_per_class
#     global all_features_class
#     results_array = []
#     startClass = 1
#     endClass = 20
#     classCombinations = combinations(range(startClass, endClass + 1), r=10)
#
#     total_cases = 0
#     total_correct = 0
#     for classCombination in classCombinations:
#         # if 10 in list(classCombination) or 26 in list(classCombination):
#         #     continue
#         # try:
#         labels = []
#         all_features = []
#         num_training_examples = 0
#
#         for class_number in classCombination:
#             num_blocks_per_class = 0
#             all_features_class = np.asarray([])
#             for filename in glob.glob(
#                     'D:/Uni/Graduation Project/All Test Cases/KHATT/Samples/Class' + str(class_number) + '/*.tif'):
#                 print(filename)
#                 temp = training(cv2.imread(filename), class_number)
#             all_features = np.append(all_features,
#                                      np.reshape(adjustNaNValues(temp), (1, num_blocks_per_class * num_features)))
#
#         # Normalization of features
#         all_features, mu, sigma = featureNormalize(np.reshape(all_features, (num_training_examples, num_features)))
#         pca = decomposition.PCA(n_components=min(all_features.shape[0], all_features.shape[1]), svd_solver='full')
#         all_features = pca.fit_transform(all_features)
#         classifier.fit(all_features, labels)
#
#         for class_number in classCombination:
#             for filename in glob.glob(
#                     'D:/Uni/Graduation Project/All Test Cases/KHATT/TestCases/testing' + str(class_number) + '.png'):
#                 print(filename)
#                 label = class_number
#                 prediction = test(cv2.imread(filename), classifier, mu, sigma, pca)
#                 total_cases += 1
#                 print(prediction[0])
#                 if prediction[0] == label:
#                     total_correct += 1
#                 results_array.append(str(prediction[0]) + '\n')
#                 print("Accuracy = ", total_correct * 100 / total_cases, " %")
#
#     results_file = open("results.txt", "w+")
#     results_file.writelines(results_array)
#     results_file.close()
#
#
# # Global Variables
# all_features = np.asarray([])
# all_features_class = np.asarray([])
# labels = []
# temp = []
# num_training_examples = 0
# num_features = 256 * 2
# num_blocks_per_class = 0
# total_test_cases = 100
#
# # classifier = MLPClassifier(solver='lbfgs', max_iter=30000, alpha=0.046041, hidden_layer_sizes=(22, 18, 15, 12, 7,),
# #                            random_state=randomState)
#
# classifier = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#                      decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
#                      max_iter=-1, probability=False, random_state=None, shrinking=True,
#                      tol=0.001, verbose=False)
# reading_test_cases()
