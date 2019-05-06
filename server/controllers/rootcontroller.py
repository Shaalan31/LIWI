import time
from flask import Flask, request, jsonify
from server.dao.connection import Database
from server.dao.writers import Writers
from texturemodel.texture_model import *
from horestmodel.horest_model import *
from siftmodel.sift_model import *
from server.httpexceptions.exceptions import *
from server.utils.writerencoder import *

app = Flask(__name__)

db = Database()
db.connect()
writers_dao = Writers()
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
    return writers_dao.get_users()


@app.route("/predict", methods=['POST'])
def get_prediction():
    try:
        if 'captured_image' in request.files:
            images = request.files['captured_image']
            filename = str(int(time.time())) + '.jpg'
            images.save(UPLOAD_FOLDER + filename)
            testing_image = cv2.imread(UPLOAD_FOLDER + filename)
            mu = 0
            sigma = 0
            t1 = threading.Thread(target=horest_model.test, args=(testing_image, mu, sigma))
            t2 = threading.Thread(target=texture_model.test, args=(testing_image, mu,sigma))
            # t3 = threading.Thread(target=self.loop_on_contours, args=(contours, 2))

            t1.start()
            t2.start()
            # t3.start()

            t1.join()
            t2.join()
            # t3.join()
    except KeyError as e:
        return 'error', 404


if __name__ == '__main__':
    app.run()
