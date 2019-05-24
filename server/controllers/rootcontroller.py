from flask import Flask, request, jsonify, send_from_directory
from server.httpexceptions.exceptions import ExceptionHandler
from server.httpresponses.errors import *
from server.utils.writerencoder import *
from server.services.writerservice import *
import uuid
import cv2

app = Flask(__name__)

writer_service = WriterService()

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


@app.route("/writers", methods=['GET'])
def get_writers_not_none():
    """
    API to get all writers for predition where features not none
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

    if language == 'en':
        status_code, message, data = writer_service.get_writers_not_none()
    else:
        status_code, message, data = writer_service.get_writers_arabic_not_none()

    raise ExceptionHandler(message=message.value, status_code=status_code.value, data=data)


@app.route("/allWriters", methods=['GET'])
def get_writers():
    """
    API to get all writers for training *Language independent
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

    status_code, message, data = writer_service.get_all_writers()

    raise ExceptionHandler(message=message.value, status_code=status_code.value, data=data)


@app.route("/predict", methods=['POST'])
def get_prediction():
    """
      API for predicting a writer of the image
      :parameter: Query parameter lang
                  - en for english
                  - ar for arabic
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
        # writers_ids = request.get_json()['writers_ids']
        language = request.args.get('lang', None)
        image_base_url = request.host_url + 'image/writers/'

        if language == "ar":
            status, message, writers_predicted = writer_service.predict_writer_arabic(testing_image, filename, image_base_url)
        else:
            status, message, writers_predicted = writer_service.predict_writer(testing_image, filename, image_base_url)

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

    status_code, message = validate_writer_request(new_writer)

    writer_id = None
    if status_code.value == 200:
        status_code, message, writer_id = writer_service.add_writer(new_writer)

    raise ExceptionHandler(message=message.value, status_code=status_code.value, data=writer_id)


@app.route("/profile", methods=['GET'])
def get_profile():
    """
    API to get writer's profile
    :parameter: Query parameter id
                Query parameter lang
                  - en for english
                  - ar for arabic
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

    status_code, message, profile_vo = writer_service.get_writer_profile(writer_id,request.host_url)

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
    :return: url for image in case found, url fo image not found in case not found
    """
    try:
        path = request.view_args['path'] + '/' + request.view_args['filename']

        return send_from_directory(UPLOAD_FOLDER, path)
    except:
        path = request.view_args['path'] + '/not_found.png'

        return send_from_directory(UPLOAD_FOLDER, path)
        # raise ExceptionHandler(message=HttpMessages.IMAGENOTFOUND.value, status_code=HttpErrors.NOTFOUND.value)


@app.route("/writer", methods=['PUT'])
def update_writer_features():
    """
    API for updating a writer features
    :parameter: Query parameter lang
                  - en for english
                  - ar for arabic
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
            status_code, message = writer_service.update_features_arabic(training_image, filename, writer_id)
        else:
            status_code, message = writer_service.update_features(training_image, filename, writer_id)

        raise ExceptionHandler(message=message.value, status_code=status_code.value)

    except KeyError as e:
        raise ExceptionHandler(message=HttpMessages.NOTFOUND.value, status_code=HttpErrors.NOTFOUND.value)


@app.route("/setWriters")
def set_writers():
    """
       API for filling database collection with dummy data
       :parameter Query parameter lang
                  - en for english
                  - ar for arabic
       :raise: Exception contains
               - response message:
                   "OK" for success
               - response status code:
                   200 for success
    """
    start_class = 609
    end_class = 2519
    language = request.args.get('lang', None)
    if language == "ar":
        # base_path = 'D:/Uni/Graduation Project/All Test Cases/KHATT/Samples/Class'
        base_path = 'C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/KHATT/Samples/Class'
        status_code, message = writer_service.fill_collection_arabic(start_class, end_class, base_path)
    else:
        base_path = 'D:/Uni/Graduation Project/All Test Cases/Dataset/Training/Class'
        # base_path = 'C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/Dataset/Training/Class'
        #Shaalan path 'C:/Users/omars/Documents/Github/LIWI/Omar/Dataset/Training/Class'S
        status_code, message = writer_service.fill_collection(start_class, end_class, base_path)

    raise ExceptionHandler(message=message.value, status_code=status_code.value)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port='5000')
