import time
from flask import Flask, request, jsonify
from server.dao.connection import Database
from server.dao.writers import Writers
from texturemodel.texture_model import *
from horestmodel.horest_model import *
from siftmodel.sift_model import *
from multiprocessing import Pool
from server.httpexceptions.exceptions import *
from server.utils.writerencoder import *
from server.models.features import *
from server.models.writer import *

app = Flask(__name__)

db = Database()
db.connect()
writers_dao = Writers(db.get_collection())
horest_model = HorestWriterIdentification()
texture_model = TextureWriterIdentification()
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.json_encoder = WriterEncoder


@app.errorhandler(ExceptionHandler)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code

    return response


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route("/users")
def get_users():
    return writers_dao.get_writers()


@app.route("/predict", methods=['POST'])
def get_prediction():
    try:
        if 'captured_image' in request.files:
            # get image from request
            images = request.files['captured_image']
            filename = str(int(time.time())) + '.jpg'
            images.save(UPLOAD_FOLDER + filename)
            testing_image = cv2.imread(UPLOAD_FOLDER + filename)

            # get features of the writers
            writers_ids = request.get_json()['writers_ids']
            writers = writers_dao.get_features(writers_ids)

            # process features to fit classifier
            label = 1
            # declaring variables for horest
            labels_horest = []
            all_features_horest = []
            num_training_examples_horest = 0

            # declaring variables for texture
            labels_texture = []
            all_features_texture = []
            num_training_examples_texture = 0

            for writer in writers:
                # processing horest_features
                horest_features = writer.feature.horest_features
                num_current_examples_horest = len(horest_features)
                labels_horest = np.append(labels_horest,
                                          np.full(shape=(1, num_current_examples_horest), fill_value=writer.id))
                num_training_examples_horest += num_current_examples_horest
                all_features_horest = np.append(all_features_horest,
                                                np.reshape(horest_features.copy(),
                                                           (1,
                                                            num_current_examples_horest * horest_model.get_num_features())))
                # processing texture_features
                texture_features = writer.feature.texture_features
                num_current_examples_texture = len(texture_features)
                labels_texture = np.append(labels_texture,
                                           np.full(shape=(1, num_current_examples_texture), fill_value=writer.id))
                num_training_examples_texture += num_current_examples_texture
                all_features_texture = np.append(all_features_texture,
                                                 np.reshape(texture_features.copy(),
                                                            (1,
                                                             num_current_examples_texture * texture_model.get_num_features())))

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

            pool = Pool(2)
            async_results = []
            async_results += [pool.apply_async(horest_model.test, (testing_image, mu_horest, sigma_horest))]
            async_results += [pool.apply_async(texture_model.test, (testing_image, mu_texture, sigma_texture, pca))]

            pool.close()
            pool.join()
            predictions = [x.get() for x in async_results]

            # used to match the probablity with classes
            horest_classes = horest_model.get_classifier_classes()
            texture_classes = texture_model.get_classifier_classes()

            print(predictions)

    except KeyError as e:
        return 'error', 404


@app.route("/setUsers")
def set_users():
    num_classes = 100
    id = 0
    names = ["", "", "", ""]
    # loop on the writers
    for class_number in range(1, num_classes + 1):
        name = names[id]
        id += 1

        writer_horest_features = np.asarray([])
        writer_texture_features = np.asarray([])
        horest_model.num_lines_per_class = 0
        texture_model.num_blocks_per_class = 0

        # loop on training data for each writer
        for filename in glob.glob(
                'D:/Uni/Graduation Project/All Test Cases/IAMJPG/Samples/Class' + str(class_number) + '/*.jpg'):
            print(filename)
            writer_horest_features = np.append(writer_horest_features, horest_model.get_features(cv2.imread(filename)))
            writer_texture_features = np.append(writer_texture_features,
                                                texture_model.get_features(cv2.imread(filename)))
            # todo calculate SDS SOH to training images per writer

        writer_horest_features = horest_model.adjust_nan_values(
            np.reshape(writer_horest_features, (horest_model.num_lines_per_class, horest_model.get_num_features())))
        writer_texture_features = texture_model.adjust_nan_values(
            np.reshape(writer_texture_features, (texture_model.num_blocks_per_class, texture_model.get_num_features())))
        # todo adjust nan for sift features

        writer = Writer()
        features = Features()
        features.horest_features = writer_horest_features
        features.texture_feature = writer_texture_features
        # Todo set sift features

        writer.features = features
        writer.id = id
        writer.name = name
        name_splitted = writer.name.split()
        writer.username = name_splitted[0][0].lower() + name_splitted[1].lower() + str(writer.id)
        status_code, message = writers_dao.create_writer(writer)
        raise ExceptionHandler(message=message.value, status_code=status_code.value)


# def addNum(num1,num2):
#     for i in range(10000000):
#         continue
#     print("in add num")
#     return num1+num2
#
# def mulNum(num1,num2):
#     print("in mul num")
#
#     return num1*num2

if __name__ == '__main__':
    app.run()
