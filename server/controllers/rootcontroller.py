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
import cv2

app = Flask(__name__)

db = Database()
db.connect()
writers_dao = Writers(db.get_collection())
horest_model = HorestWriterIdentification()
texture_model = TextureWriterIdentification()
sift_model = SiftModel()
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


@app.route("/writers")
def get_writers():
    status_code, message, data = writers_dao.get_writers()

    raise ExceptionHandler(message=message.value, status_code=status_code.value, data=data)


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


@app.route("/setWriters")
def set_writers():
    num_classes = 100
    id = 0
    names = ["Abdul Ahad", "Abdul Ali",
             "Abdul Alim", "Abdul Azim",
             "Abu Abdullah", "Abu Hamza",
             "Ahmed Tijani", "Ali Reza",
             "Aman Ali", "Anisur Rahman",
             "Azizur Rahman", "Badr al-Din",
             "Baha' al-Din", "Barkat Ali",
             "Burhan al-Din", "Fakhr al-Din",
             "Fazl UrRahman", "Fazlul Karim",
             "Fazlul Haq", "Ghulam Faruq",
             "Ghiyath al-Din", "Ghulam Mohiuddin",
             "Habib ElRahman", "Hamid al-Din",
             "Hibat Allah", "Husam ad-Din",
             "Ikhtiyar al-Din", "Imad al-Din",
             "Izz al-Din", "Jalal ad-Din",
             "Jamal ad-Din", "Kamal ad-Din",
             "Lutfur Rahman", "Mizanur Rahman",
             "Mohammad Taqi", "Nasir al-Din",
             "Seif ilislam", "Sadr al-Din",
             "Sddam Hussein", "Samar Gamal",
             "May Ahmed", "Ahmed Khairy",
             "Omar Ali", "Salma Ibrahim",
             "Ahmed Gamal", "Hadeer Hossam",
             "Hanaa Ahmed", "Gamal Saad",
             "Bisa Dewidar", "Ahmed Said",
             "Nachwa Ahmed", "Ezz Farhan",
             "Nourhan Farhan", "Mariam Farhan",
             "Mouhab Farhan", "Sherif Ahmed",
             "Noha Ahmed", "Yasmine Sherif",
             "Eslam Sherif", "Ahmed Sherif",
             "Mohamed Ahmed", "Zeinab Khairy",
             "Khaled Ali", "Rana Ali",
             "Ali Shaalan", "Ahmed Youssry",
             "AbdelRahman Nasser", "Youssra Hussein",
             "Ingy Alaa", "Rana Afifi",
             "Nour Attya", "Amani Tarek",
             "Salma Ahmed", "Iman Fouad",
             "Karim ElRashidy", "Ziad Mansour",
             "Mohamed Salah", "Anas ElShazly",
             "Hazem Aly", "Youssef Maraghy",
             "Ebram Hossam", "Mohamed Nour",
             "Mohamed Ossama", "Hussein Hosny",
             "Ahmed Samy", "Youmna Helmy",
             "Kareem Haggag", "Nour Yasser",
             "Farah Mohamed", "Ahmed Hisham",
             "Omar Nashaat", "Mohamed Yasser",
             "Sara Hassan", "Ahmed keraidy",
             "Magdy Hafez", "Waleed Mostafa",
             "Khaled Hesham", "Karim Hossam",
             "Omar Nasharty", "Rayhana Ayman"]
    # loop on the writers
    for class_number in range(1, num_classes + 1):
        writer_name = names[id]
        id += 1

        writer_horest_features = []
        writer_texture_features = []
        SDS_train = []
        SOH_train = []
        horest_model.num_lines_per_class = 0
        texture_model.num_blocks_per_class = 0
        print('Class' + str(class_number) + ':')

        # loop on training data for each writer
        for filename in glob.glob(
                'C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/TestCasesCompressed/Samples/Class' + str(class_number) + '/*.jpg'):
            print(filename)
            print('Horest Features')
            # writer_horest_features.append(horest_model.get_features(cv2.imread(filename))[0].tolist())
            writer_horest_features = np.append(writer_horest_features, horest_model.get_features(cv2.imread(filename))[0].tolist())
            print('Texture Features')
            # writer_texture_features.append(texture_model.get_features(cv2.imread(filename))[0].tolist())
            writer_texture_features = np.append(writer_texture_features, texture_model.get_features(cv2.imread(filename))[0].tolist())
            print('Sift Model')
            name = Path(filename).name
            SDS, SOH = sift_model.get_features(filename, name)
            SDS_train.append(SDS[0].tolist())
            SOH_train.append(SOH[0].tolist())

        writer_horest_features = horest_model.adjust_nan_values(
            np.reshape(writer_horest_features, (horest_model.num_lines_per_class, horest_model.get_num_features()))).tolist()
        writer_texture_features = texture_model.adjust_nan_values(
            np.reshape(writer_texture_features, (texture_model.num_blocks_per_class, texture_model.get_num_features()))).tolist()

        writer = Writer()
        features = Features()
        features.horest_features = writer_horest_features
        features.texture_feature = writer_texture_features
        features.sift_SDS = SDS_train
        features.sift_SOH = SOH_train

        writer.features = features
        writer.id = id
        writer.name = writer_name
        name_splitted = writer.name.split()
        writer.username = name_splitted[0][0].lower() + name_splitted[1].lower() + str(writer.id)
        status_code, message = writers_dao.create_writer(writer)
        print(message.value)

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
