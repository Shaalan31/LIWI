from flask import Flask, request, jsonify, send_from_directory
from server.dao.connection import Database
from server.dao.writers import Writers
from server.httpexceptions.exceptions import ExceptionHandler
from server.utils.writerencoder import *
from server.utils.utilities import *
from server.services.writerservice import *

import uuid
import cv2

app = Flask(__name__)

db = Database()
db.connect()
db.create_collection()
writers_dao = Writers(db.get_collection())

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../../uploads/')
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
    language = request.args.get('lang', None)
    dao = writers_dao
    if language == "ar":
        dao = Writers(db.get_collection_arabic())

    status_code, message, data = dao.get_writers()

    raise ExceptionHandler(message=message.value, status_code=status_code.value, data=data)


@app.route("/predict", methods=['POST'])
def get_prediction():
    """
      API for predicting a writer of the image
      :parameter: request contains
                  - writers ids: writers_ids
                  - image name: _filename
      :raise: Exception contains
              - response message:
                  "OK" for success, "Error in prediction" for prediction conflict,"Maximum number of writers exceeded" for exceeding maximum numbers
              - response status code:
                  200 for success, 500 for prediction conflict,400 for exceeding maximum number
      """
    print("New prediction request")
    try:
        # get image from request
        filename = request.get_json()['_filename']
        testing_image = cv2.imread(UPLOAD_FOLDER + 'testing/' + filename)

        # get features of the writers
        writers_ids = request.get_json()['writers_ids']
        language = request.args.get('lang', None)
        image_base_url = request.host_url + 'image/writers/'

        if language == "ar":
            status, message, writers_predicted = predict_writer_arabic(testing_image, filename, writers_ids,
                                                                       Writers(db.get_collection_arabic()),
                                                                       image_base_url)
        else:
            status, message, writers_predicted = predict_writer(testing_image, filename, writers_ids, writers_dao,
                                                                image_base_url)

        raise ExceptionHandler(message=message.value, status_code=status.value,
                               data=writers_predicted)
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
    language = request.args.get('lang', None)
    dao = writers_dao
    if language == "ar":
        dao = Writers(db.get_collection_arabic())

    status_code, message = validate_writer_request(new_writer)
    if status_code.value == 200:
        writer = Writer()
        writer.name = new_writer["_name"]
        writer.username = new_writer["_username"]
        writer.address = new_writer["_address"]
        writer.phone = new_writer["_phone"]
        writer.nid = new_writer["_nid"]
        writer.image = new_writer["_image"]
        writer.id = dao.get_writers_count() + 1

        status_code, message = dao.create_writer(writer)

    raise ExceptionHandler(message=message.value, status_code=status_code.value)


@app.route("/profile", methods=['GET'])
def get_profile():
    """
    API to get writer's profile
    :parameter: query parameter id
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

    language = request.args.get('lang', None)
    dao = writers_dao
    if language == "ar":
        dao = Writers(db.get_collection_arabic())

    status_code, message, profile_vo = dao.get_writer_profile(writer_id)

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

        raise ExceptionHandler(message=HttpMessages.SUCCESS.value, status_code=HttpErrors.SUCCESS.value,
                               data=image_name)
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
    """
    API for updating a writer features
    :parameter: request contains
                - image name: _filename
                - writer id: _id
    :raise: Exception contains
            - response message:
                "OK" for success, "Not found" for image not found
            - response status code:
                200 for success, 400 for image not found
    """
    try:
        # get image from request
        filename = request.get_json()['_filename']
        training_image = cv2.imread(UPLOAD_FOLDER + 'training/' + filename)

        # get writer
        writer_id = int(request.get_json()['_id'])

        language = request.args.get('lang', None)
        if language == "ar":
            status_code, message = update_features_arabic(training_image, filename, writer_id,
                                                          Writers(db.get_collection_arabic()))
        else:
            status_code, message = update_features(training_image, filename, writer_id, writers_dao)

        raise ExceptionHandler(message=message.value, status_code=status_code.value)

    except KeyError as e:
        raise ExceptionHandler(message=HttpMessages.NOTFOUND.value, status_code=HttpErrors.NOTFOUND.value)


@app.route("/setWriters")
def set_writers():
    """
       API for filling database collection with dummy data
       :raise: Exception contains
               - response message:
                   "OK" for success
               - response status code:
                   200 for success
       """
    start_class = 1
    end_class = 100
    language = request.args.get('lang', None)
    if language == "ar":
        base_path = 'D:/Uni/Graduation Project/All Test Cases/KHATT/Samples/Class'
        status_code, message = fill_collection_arabic(start_class, end_class, base_path,
                                                      Writers(db.get_collection_arabic()))
    else:
        base_path = 'D:/Uni/Graduation Project/All Test Cases/IAMJPG/Samples/Class'
        status_code, message = fill_collection(start_class, end_class, base_path, writers_dao)

    raise ExceptionHandler(message=message.value, status_code=status_code.value)


if __name__ == '__main__':
    app.run(host='192.168.1.11', port='5000')
