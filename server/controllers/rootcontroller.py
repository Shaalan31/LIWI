import time
from flask import Flask, request
from server.dao.connection import Database
from server.dao.writers import Writers
from texturemodel.texture_model import *
from horestmodel.horest_model import *
from siftmodel.sift_model import *
from multiprocessing import Pool

app = Flask(__name__)

db = Database()
db.connect()
writers_dao = Writers(db.get_collection())
horest_model = HorestWriterIdentification()
texture_model = TextureWriterIdentification()
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route("/users")
def get_users():
    return writers_dao.get_users()


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
            features = writers_dao.get_features(writers_ids)

            # process features to fit classifier
            label = 1
            # declaring variables for horest
            labels_horest = []
            all_features_horest = []
            num_training_examples_horest = 0
            num_features_horest = 18

            # declaring variables for texture
            labels_texture = []
            all_features_texture = []
            num_training_examples_texture = 0
            num_features_texture = 256 * 2

            for feature in features:
                # processing horest_features
                horest_features = feature.horest_features
                num_current_examples_horest = len(horest_features)
                labels_horest = np.append(labels_horest,
                                          np.full(shape=(1, num_current_examples_horest), fill_value=label))
                num_training_examples_horest += num_current_examples_horest
                all_features_horest = np.append(all_features_horest,
                                                np.reshape(horest_features.copy(),
                                                           (1, num_current_examples_horest * num_features_horest)))
                # processing texture_features
                texture_features = feature.texture_features
                num_current_examples_texture = len(texture_features)
                labels_texture = np.append(labels_texture,
                                           np.full(shape=(1, num_current_examples_texture), fill_value=label))
                num_training_examples_texture += num_current_examples_texture
                all_features_texture = np.append(all_features_texture,
                                                 np.reshape(texture_features.copy(),
                                                            (1, num_current_examples_texture * num_features_texture)))

            # fit horest classifier
            all_features_horest = np.reshape(all_features_horest, (num_training_examples_horest, num_features_horest))
            all_features_horest, mu_horest, sigma_horest = horest_model.feature_normalize(all_features_horest)
            horest_model.fit_classifier(all_features_horest, labels_horest)

            # fit texture classifier
            all_features_texture = np.reshape(all_features_texture,
                                              (num_training_examples_texture, num_features_texture))
            all_features_texture, mu_texture, sigma_texture = texture_model.feature_normalize(all_features_texture)
            texture_model.fit_classifier(all_features_texture, labels_texture)

            pool = Pool(2)
            async_results = []
            async_results += [pool.apply_async(horest_model.test, (testing_image, mu_horest, sigma_horest))]
            async_results += [pool.apply_async(texture_model.test, (testing_image, mu_texture, sigma_texture))]

            pool.close()
            pool.join()
            results = [x.get() for x in async_results]
            print(results)

    except KeyError as e:
        return 'error', 404


#
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
