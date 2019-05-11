from server.httpresponses.errors import HttpErrors
from server.httpresponses.messages import HttpMessages

from texturemodel.texture_model import *
from horestmodel.horest_model import *
from siftmodel.sift_model import *
from server.views.writervo import *

from multiprocessing import Pool

horest_model = HorestWriterIdentification()
texture_model = TextureWriterIdentification()
sift_model = SiftModel()


def predict_writer(testing_image, filename, writers_ids, dao):
    """
        Service to get predicted English writers
        :parameter: request contains
                  - testing image: testing_image
                  - image name: filename
                  - writers ids: writers_ids
                  - dao: dao
        :return:
                -status
                -message
                -writers_predicted
      """

    if len(writers_ids) > 30:
        return HttpErrors.BADREQUEST, HttpMessages.MAXIMUM_EXCEEDED, None

    writers = dao.get_features(writers_ids)

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

    for writer in writers:
        # processing horest_features
        horest_features = writer.features.horest_features
        num_current_examples_horest = len(horest_features)
        labels_horest = np.append(labels_horest,
                                  np.full(shape=(1, num_current_examples_horest), fill_value=writer.id))
        num_training_examples_horest += num_current_examples_horest
        all_features_horest = np.append(all_features_horest,
                                        np.reshape(horest_features.copy(),
                                                   (1,
                                                    num_current_examples_horest * horest_model.get_num_features())))
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

    pool = Pool(3)
    async_results = []
    async_results += [pool.apply_async(horest_model.test, (testing_image, mu_horest, sigma_horest))]
    async_results += [pool.apply_async(texture_model.test, (testing_image, mu_texture, sigma_texture, pca))]
    async_results += [pool.apply_async(sift_model.predict, (SDS_train, SOH_train, testing_image, filename))]

    pool.close()
    pool.join()

    predictions = []
    # used to match the probablity with classes
    horest_classes = horest_model.get_classifier_classes()
    horest_predictions = async_results[0].get()[0]
    predictions.append(horest_classes[np.argmax(horest_predictions)])
    horest_indecies_sorted = np.argsort(horest_classes, axis=0)
    sorted_horest_classes = horest_classes[horest_indecies_sorted[::-1]]
    sorted_horest_predictions = horest_predictions[horest_indecies_sorted[::-1]]

    texture_classes = texture_model.get_classifier_classes()
    texture_predictions = async_results[1].get()[0]
    predictions.append(texture_classes[np.argmax(texture_predictions)])
    texture_indecies_sorted = np.argsort(texture_classes, axis=0)
    sorted_texture_predictions = texture_predictions[texture_indecies_sorted[::-1]]

    score = sorted_horest_predictions + sorted_texture_predictions

    sift_prediction = async_results[2].get()
    sift_prediction = writers_lookup_array[sift_prediction]
    predictions.append(sift_prediction)

    score[np.argwhere(sift_prediction)] += (1 / 3)

    final_prediction = int(sorted_horest_classes[np.argmax(score)])

    writers_predicted = []
    vfunc = np.vectorize(func)
    writer_predicted = writers[np.where(vfunc(writers) == final_prediction)[0][0]]
    writer_vo = WriterVo(writer_predicted.id, writer_predicted.name, writer_predicted.username)
    writers_predicted.append(writer_vo)

    predictions = np.unique(predictions)
    if len(predictions) > 1:
        for prediction in predictions:
            if prediction != final_prediction:
                writer_predicted = writers[np.where(vfunc(writers) == prediction)[0][0]]
                writer_vo = WriterVo(writer_predicted.id, writer_predicted.name, writer_predicted.username)
                writers_predicted.append(writer_vo)

    return HttpErrors.SUCCESS, HttpMessages.OK, writers_predicted


def predict_writer_arabic(testing_image, filename, writers_ids, dao):
    """
         Service to get predicted Arabic writers
         :parameter: request contains
                   - testing image: testing_image
                   - image name: filename
                   - writers ids: writers_ids
                   - dao: dao
         :return:
                 -status
                 -message
                 -writers_predicted
   """

    if len(writers_ids) > 3:
        return HttpErrors.BADREQUEST, HttpMessages.MAXIMUM_EXCEEDED, None

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
        # for i in range(len(writer.features.sift_SDS)):
        #     SDS_train.append(np.array([writer.features.sift_SDS[i]]))
        #     SOH_train.append(np.array([writer.features.sift_SOH[i]]))
        #     writers_lookup_array.append(writer.id)

    # fit texture classifier
    all_features_texture = np.reshape(all_features_texture,
                                      (num_training_examples_texture, texture_model.get_num_features()))
    all_features_texture, mu_texture, sigma_texture = texture_model.feature_normalize(all_features_texture)
    pca = decomposition.PCA(n_components=min(all_features_texture.shape[0], all_features_texture.shape[1]),
                            svd_solver='full')
    all_features_texture = pca.fit_transform(all_features_texture)
    texture_model.fit_classifier(all_features_texture, labels_texture)

    # pool = Pool(2)
    pool = Pool(1)
    async_results = []
    async_results += [pool.apply_async(texture_model.test, (testing_image, mu_texture, sigma_texture, pca))]
    # async_results += [pool.apply_async(sift_model.predict, (SDS_train, SOH_train, testing_image, filename))]

    pool.close()
    pool.join()

    predictions = []
    # used to match the probability with classes
    texture_classes = texture_model.get_classifier_classes()
    texture_predictions = async_results[1].get()[0]
    predictions.append(texture_classes[np.argmax(texture_predictions)])
    texture_indecies_sorted = np.argsort(texture_classes, axis=0)
    sorted_texture_predictions = texture_predictions[texture_indecies_sorted[::-1]]
    sorted_texture_classes = texture_classes[texture_indecies_sorted[::-1]]

    score = sorted_texture_predictions

    # sift_prediction = async_results[2].get()
    # sift_prediction = writers_lookup_array[sift_prediction]
    # predictions.append(sift_prediction)

    # score[np.argwhere(sift_prediction)] += (1 / 3)

    final_prediction = int(sorted_texture_classes[np.argmax(score)])

    writers_predicted = []
    vfunc = np.vectorize(func)
    writer_predicted = writers[np.where(vfunc(writers) == final_prediction)[0][0]]
    writer_vo = WriterVo(writer_predicted.id, writer_predicted.name, writer_predicted.username)
    writers_predicted.append(writer_vo)

    predictions = np.unique(predictions)
    if len(predictions) > 1:
        for prediction in predictions:
            if prediction != final_prediction:
                writer_predicted = writers[np.where(vfunc(writers) == prediction)[0][0]]
                writer_vo = WriterVo(writer_predicted.id, writer_predicted.name, writer_predicted.username)
                writers_predicted.append(writer_vo)

    return HttpErrors.SUCCESS, HttpMessages.OK, writers_predicted


def update_features(training_image, filename, writer_id, dao):
    """
             Service to get predicted English writers
             :parameter: request contains
                       - training image: training_image
                       - image name: filename
                       - writer id: writer_id
                       - dao: dao
             :return:
                     -status
                     -message
   """
    writer = dao.get_writer(writer_id)

    # get features
    pool = Pool(3)
    async_results = []
    async_results += [pool.apply_async(horest_model.get_features, (training_image,))]
    async_results += [pool.apply_async(texture_model.get_features, (training_image,))]
    async_results += [pool.apply_async(sift_model.get_features, (filename, training_image))]

    pool.close()
    pool.join()
    num_lines, horest_features = async_results[0].get()
    num_blocks, texture_features = async_results[1].get()
    SDS, SOH = async_results[2].get()

    # adjust nan
    horest_features = horest_model.adjust_nan_values(
        np.reshape(horest_features,
                   (num_lines, horest_model.get_num_features()))).tolist()
    texture_features = texture_model.adjust_nan_values(
        np.reshape(texture_features,
                   (num_blocks,
                    texture_model.get_num_features()))).tolist()

    # set features
    features = writer.features
    features.horest_features.extend(horest_features)
    features.texture_feature.extend(texture_features)

    features.sift_SDS.append(SDS[0].tolist())
    features.sift_SOH.append(SOH[0].tolist())

    status_code, message = dao.update_writer(writer)
    return status_code, message


def update_features_arabic(training_image, filename, writer_id, dao):
    """
         Service to get predicted English writers
         :parameter: request contains
                   - training image: training_image
                   - image name: filename
                   - writer id: writer_id
                   - dao: dao
         :return:
                 -status
                 -message
   """
    writer = dao.get_writer(writer_id)

    # get features
    pool = Pool(1)
    # pool = Pool(2)
    async_results = []
    async_results += [pool.apply_async(texture_model.get_features, (training_image,))]
    # async_results += [pool.apply_async(sift_model.get_features, (filename, training_image))]

    pool.close()
    pool.join()

    num_blocks, texture_features = async_results[1].get()
    # SDS, SOH = async_results[2].get()

    # adjust nan
    texture_features = texture_model.adjust_nan_values(
        np.reshape(texture_features,
                   (num_blocks,
                    texture_model.get_num_features()))).tolist()

    # set features
    features = writer.features
    features.texture_feature.extend(texture_features)

    # features.sift_SDS.append(SDS[0].tolist())
    # features.sift_SOH.append(SOH[0].tolist())

    status_code, message = dao.update_writer(writer)
    return status_code, message


def func(writer):
    """
    Function to return attribute writer id
    :param writer: writer model object
    :return: attribute id
    """
    return writer.id
