import time
from flask import Flask, request, jsonify, send_from_directory
from server.dao.connection import Database
from server.dao.writers import Writers
from texturemodel.texture_model import *
from horestmodel.horest_model import *
from siftmodel.sift_model import *
from multiprocessing import Pool
from server.httpexceptions.exceptions import *
from server.utils.writerencoder import *
from server.views.writervo import *
from server.utils.utilities import *
import uuid
import cv2

app = Flask(__name__)

db = Database()
db.connect()
writers_dao = Writers(db.get_collection())
horest_model = HorestWriterIdentification()
texture_model = TextureWriterIdentification()
sift_model = SiftModel()
UPLOAD_FOLDER = 'C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.json_encoder = WriterEncoder


@app.errorhandler(ExceptionHandler)
def handle_invalid_usage(error):
    """
    Error Handler for class Exception Handler
    :param error:
    :return: response containing:
             status code, message, and data
    """
    response = jsonify(error.to_dict())
    response.status_code = error.status_code

    return response


@app.route("/writers")
def get_writers():
    """
    API to get all writers
    :raise: Exception containing:
            message:
            - "OK" for success
            - "No writers found"  if there is no writer
            status_code:
            - 200 for success
            - 404 if there is no writer
            data:
            - list of WritersVo: each writervo contains id, name, username
            - None if there is no writer
    """
    status_code, message, data = writers_dao.get_writers()

    raise ExceptionHandler(message=message.value, status_code=status_code.value, data=data)


@app.route("/predict", methods=['POST'])
def get_prediction():
    print("New prediction request")
    try:
        if 'captured_image' in request.files:
            # get image from request
            images = request.files['captured_image']
            filename = str(int(time.time())) + '.jpg'
            images.save(UPLOAD_FOLDER + filename)
            testing_image = cv2.imread(UPLOAD_FOLDER + filename)

            # get features of the writers
            writers_ids = request.form['writers_ids']
            writers_ids = list(map(int, writers_ids[1:len(writers_ids) - 1].split(',')))
            writers = writers_dao.get_features(writers_ids)

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

            # used to match the probablity with classes
            horest_classes = horest_model.get_classifier_classes()
            horest_predictions = async_results[0].get()[0]
            horest_indecies_sorted = np.argsort(horest_classes, axis=0)
            sorted_horest_classes = horest_classes[horest_indecies_sorted[::-1]]
            sorted_horest_predictions = horest_predictions[horest_indecies_sorted[::-1]]

            texture_classes = texture_model.get_classifier_classes()
            texture_predictions = async_results[1].get()[0]
            texture_indecies_sorted = np.argsort(texture_classes, axis=0)
            sorted_texture_predictions = texture_predictions[texture_indecies_sorted[::-1]]

            score = sorted_horest_predictions + sorted_texture_predictions
            prediction_horest_texture = sorted_horest_classes[np.argmax(score)]
            print(prediction_horest_texture)

            sift_prediction = async_results[2].get()
            sift_prediction = writers_lookup_array[sift_prediction]
            score[np.argwhere(sift_prediction)] += (1 / 3)
            print(sift_prediction)

            final_prediction = int(sorted_horest_classes[np.argmax(score)])
            print(final_prediction)

            vfunc = np.vectorize(func)
            writer_predicted = writers[np.where(vfunc(writers) == final_prediction)[0][0]]
            writer_vo = WriterVo(writer_predicted.id, writer_predicted.name, writer_predicted.username)
        raise ExceptionHandler(message=HttpMessages.SUCCESS.value, status_code=HttpErrors.SUCCESS.value, data=writer_vo)
    except KeyError as e:
        raise ExceptionHandler(message=HttpMessages.CONFLICT_PREDICTION.value, status_code=HttpErrors.CONFLICT.value)


@app.route("/writer", methods=['POST'])
def create_writer():
    """
    API for creating a new writer
    :parameter: request contains
                - writer name: _name
                - writer username: _username
                - image name: _image
                - address: _address
                - phone: _phone
                - national id: _nid
    :raise: Exception contains
            - response message:
                "OK" for success, "Writer already exists" for duplicate username
            - response status code:
                200 for success, 409 for duplicate username
    """
    # request parameters
    new_writer = request.get_json()

    status_code, message = validate_writer_request(new_writer)
    if status_code.value == 200:
        writer = Writer()
        writer.name = new_writer["_name"]
        writer.username = new_writer["_username"]
        writer.address = new_writer["_address"]
        writer.phone = new_writer["_phone"]
        writer.nid = new_writer["_nid"]
        writer.image = new_writer["_image"]
        writer.id = writers_dao.get_writers_count() + 1

        status_code, message = writers_dao.create_writer(writer)

    raise ExceptionHandler(message=message.value, status_code=status_code.value)


@app.route("/profile", methods=['GET'])
def get_profile():
    """
    API to get writer's profile
    :raise: Exception containing:
            message:
            - "OK" for success
            - "Writer is not found" if writer does not exist
            status_code:
            - 200 for success
            - 404 if writer does not exist
            data:
            - ProfileVo object containing writer's: id, name, username, address, phone, nid
            - None if writer does not exist
    """
    writer_id = request.args.get('id', None)

    status_code, message, profile_vo = writers_dao.get_writer_profile(writer_id)

    raise ExceptionHandler(message=message.value, status_code=status_code.value, data=profile_vo)


@app.route("/image/<path>", methods=['POST'])
def upload_image(path):
    """
    API for uploading images
    request: image: file of the image
    :param: path: path variable to identify the folder to upload in
            - writers: for writers
            - testing: for testing
            - training: for training
    :raise: Exception contains
            - response message:
                "OK" for success, "Upload image failed" for any fail in upload
            - response status code:
                200 for success, 409 for any fail in upload
    """
    try:
        path = request.view_args['path']
        image = request.files["image"]
        image_name = str(uuid.uuid1()) + '.jpg'
        image.save(UPLOAD_FOLDER + path + '/' + image_name)

        raise ExceptionHandler(message=HttpMessages.SUCCESS.value, status_code=HttpErrors.SUCCESS.value, data=image_name)
    except KeyError as e:
        raise ExceptionHandler(message=HttpMessages.UPLOADFAIL.value, status_code=HttpErrors.CONFLICT.value)


@app.route("/image/<path>/<filename>", methods=['GET'])
def get_image(path, filename):
    """
    API to get the image
    :param path: path variable for folder to get the image from
                - writers: for writers
                - testing: for testing
                - training: for training
    :param filename: path variable for image name
    :return:
    """
    try:
        path = request.view_args['path'] + '/' + request.view_args['filename']

        return send_from_directory(UPLOAD_FOLDER, path)
    except:
        raise ExceptionHandler(message=HttpMessages.IMAGENOTFOUND.value, status_code=HttpErrors.NOTFOUND.value)


@app.route("/writer", methods=['PUT'])
def update_writer_features():
    try:
        if 'captured_image' in request.files:
            # get image from request
            images = request.files['captured_image']
            filename = str(int(time.time())) + '.jpg'
            images.save(UPLOAD_FOLDER + filename)
            training_image = cv2.imread(UPLOAD_FOLDER + filename)

            # get writer
            writers_id = int(request.form['id'])
            writer = writers_dao.get_writer(writers_id)

            pool = Pool(3)
            async_results = []
            async_results += [pool.apply_async(horest_model.get_features, (training_image))]
            async_results += [pool.apply_async(texture_model.get_features, (training_image))]
            async_results += [pool.apply_async(sift_model.get_features, (filename,training_image))]

            pool.close()
            pool.join()

            status_code, message = writers_dao.update_writer(writer)
    except KeyError as e:
        raise ExceptionHandler(message=HttpMessages.NOTFOUND.value, status_code=HttpErrors.NOTFOUND.value)


@app.route("/setWriters")
def set_writers():
    num_classes = 100

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
    for class_number in range(70, num_classes + 1):
        writer_name = names[class_number - 1]

        writer_horest_features = []
        writer_texture_features = []
        SDS_train = []
        SOH_train = []
        horest_model.num_lines_per_class = 0
        texture_model.num_blocks_per_class = 0
        print('Class' + str(class_number) + ':')

        # loop on training data for each writer
        for filename in glob.glob(
                'D:/Uni/Graduation Project/All Test Cases/IAMJPG/Samples/Class' + str(class_number) + '/*.jpg'):
            print(filename)
            image = cv2.imread(filename)
            print('Horest Features')
            # writer_horest_features.append(horest_model.get_features(cv2.imread(filename))[0].tolist())
            writer_horest_features = np.append(writer_horest_features, horest_model.get_features(image)[0].tolist())
            print('Texture Features')
            # writer_texture_features.append(texture_model.get_features(cv2.imread(filename))[0].tolist())
            writer_texture_features = np.append(writer_texture_features, texture_model.get_features(image)[0].tolist())
            print('Sift Model')
            name = Path(filename).name
            SDS, SOH = sift_model.get_features(name, image=image)
            SDS_train.append(SDS[0].tolist())
            SOH_train.append(SOH[0].tolist())

        writer_horest_features = horest_model.adjust_nan_values(
            np.reshape(writer_horest_features,
                       (horest_model.num_lines_per_class, horest_model.get_num_features()))).tolist()
        writer_texture_features = texture_model.adjust_nan_values(
            np.reshape(writer_texture_features,
                       (texture_model.num_blocks_per_class, texture_model.get_num_features()))).tolist()

        writer = Writer()
        features = Features()
        features.horest_features = writer_horest_features
        features.texture_feature = writer_texture_features
        features.sift_SDS = SDS_train
        features.sift_SOH = SOH_train

        writer.features = features
        writer.id = class_number
        writer.name = writer_name
        name_splitted = writer.name.split()
        writer.username = name_splitted[0][0].lower() + name_splitted[1].lower() + str(writer.id)
        status_code, message = writers_dao.create_writer(writer)
        print(message.value)

    raise ExceptionHandler(message=message.value, status_code=status_code.value)


if __name__ == '__main__':
    app.run()
