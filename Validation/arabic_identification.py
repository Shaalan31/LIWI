import pickle
from siftmodel.sift_model_test import *
import pickle
from texturemodel.texture_model import *


#hyperparameters
t_test=50
phi_test=36
w_test=0.5
h_coeff=0.7
code_book = pickle.load( open( "centers_KHATT.pkl", "rb" ) )








texture_model = TextureWriterIdentification('D:/Uni/Graduation Project/All Test Cases/KHATT/Samples/Class',
                                            # 'D:/Uni/Graduation Project/All Test Cases/KHATT/TestCases/testing')
                                            'C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/KHATT/TestCases/testing')





def preprocess_texture(h,list_classes):
    # print("Preprocessing")
    base_samples_h = 'C:/Users/omars/Documents/Github/LIWI/Omar/ValidationArabic/Samples/H/'

    h_coeff = str(h)
    # declaring variables for texture
    labels_texture = []
    all_features_texture = []
    num_training_examples_texture = 0


    for writer_id in list_classes:
        for filename in glob.glob(base_samples_h+str(h_coeff)+"/Class"+str(writer_id)+'/*.csv'):
            # print(filename)
            # texture_features = writer.features.texture_feature
            texture_features = np.genfromtxt(filename,delimiter=',')
            num_current_examples_texture = len(texture_features)
            labels_texture = np.append(labels_texture,
                                       np.full(shape=(1, num_current_examples_texture), fill_value=writer_id))
            num_training_examples_texture += num_current_examples_texture
            all_features_texture = np.append(all_features_texture,
                                             np.reshape(texture_features.copy(),
                                                        (1,
                                                         num_current_examples_texture * texture_model.get_num_features())))


    # fit texture classifier
    all_features_texture = np.reshape(all_features_texture,
                                      (num_training_examples_texture, texture_model.get_num_features()))

    all_features_texture, mu_texture, sigma_texture = texture_model.feature_normalize(all_features_texture)
    pca = decomposition.PCA(n_components=min(all_features_texture.shape[0], all_features_texture.shape[1]),
                            svd_solver='full')
    # all_features_texture.dropna(inplace=True)
    all_features_texture = pca.fit_transform(all_features_texture)
    texture_model.fit_classifier_arabic(all_features_texture, labels_texture)

    return mu_texture, sigma_texture, pca


def predict_texture(testing_image, mu_texture, sigma_texture, pca,sift_prediction,prob_sift):
    texture_features = np.genfromtxt(testing_image,delimiter=',')
    # print("Starting Texture Testing")
    texture_classes = texture_model.get_classifier_classes_arabic()
    texture_predictions = texture_model.fast_test(texture_features, mu_texture, sigma_texture,pca,lang='ar')[0]

    texture_indecies_sorted = np.argsort(texture_classes, axis=0)
    sorted_texture_predictions = texture_predictions[texture_indecies_sorted[::-1]]
    sorted_texture_classes = texture_classes[texture_indecies_sorted[::-1]]

    score = (1 - float(prob_sift)) * sorted_texture_predictions
    score[np.argwhere(sorted_texture_classes == sift_prediction)] += float(prob_sift)
    final_prediction = int(sorted_texture_classes[np.argmax(score)])
    # print("Texture Prediction:" + str(sorted_texture_classes[np.argmax(sorted_texture_predictions)]))

    # score = 0.25 * sorted_horest_predictions + 0.25 * sorted_texture_predictions

    # print("Starting Sift Testing")
    # sift_prediction = sift_model.predict(SDS_train, SOH_train, testing_image, filename,lang="en")
    # sift_prediction = writers_lookup_array[sift_prediction]
    # print("Sift Prediction:" + str(sift_prediction))

    # score[np.argwhere(sorted_texture_classes == sift_prediction)] += (1 / 2)
    # final_prediction = int(sorted_horest_classes[np.argmax(score)])
    final_prediction = sorted_texture_classes[np.argmax(sorted_texture_predictions)]
    # print("Common Prediction: " + str(final_prediction))

    return final_prediction


start = 1
end = 120
prob_sift = [0.1,0.5,0.9]
acc = None

for radius in range(50,121,10):
    class_labels = list(range(start, end))
    classCombinations = combinations(class_labels, r=radius)#end - start)
    right_test_cases = 0
    total_test_cases = 0
    accuracy = 0

    # print(h_coeff)

    for prob_sift_test in prob_sift:
        testcases = 0
        for item in classCombinations:
            print(testcases)
            # try:
            mu_texture, sigma_texture, pca = preprocess_texture(h_coeff, item)
            sift_model = SiftModel(test_classes=item, code_book=code_book, t=t_test, phi=phi_test, w=w_test,lang='ar')
            sift_model.run()
            sift_prediction = sift_model.prediction
            i = 0
            for count in item:
                # print('Class' + str(count) + ':')
                for filename in glob.glob('C:/Users/omars/Documents/Github/LIWI/Omar/ValidationArabic/TestCases/H/'+str(h_coeff)+ '/testing' + str(count) + '.csv'):
                    name = Path(filename).name
                    # print(name)
                    # image = cv2.imread(filename)
                    prediction = predict_texture(filename, mu_texture, sigma_texture, pca,sift_prediction[i],prob_sift_test)
                    i+=1

                    if (prediction == count):
                        right_test_cases += 1
                    total_test_cases += 1
                    testcases += 1
                    accuracy = (right_test_cases / total_test_cases) * 100

                    print("Accuracy: " + str(accuracy) + "%")
            # except:
            #     pass
            if testcases >= 20:
                if acc is None:
                    acc = np.array([radius, prob_sift_test, accuracy]).reshape((1, 3))
                else:
                    acc = np.append(acc,np.array([radius, prob_sift_test, accuracy]).reshape((1,3)),axis=0)
                print('Acc finaal @ p=', prob_sift_test, ' - ', accuracy, 'rad - ', radius)
                print('shape ',acc.shape)
                np.savetxt('ar_results.csv',acc,delimiter=',')
                break

