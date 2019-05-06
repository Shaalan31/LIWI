from server.dao.connection import *
from server.dao.writers import *
from server.models.writer import *
from server.httpexceptions.exceptions import *
from server.utils.writerencoder import *
from flask import jsonify


from flask import Flask
app = Flask(__name__)
app.json_encoder = WriterEncoder

@app.errorhandler(ExceptionHandler)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code

    return response

# test for create, update, get writer, get features
@app.route('/')
def test():
    # Create writer
    db = Database()
    db.connect()

    writers = Writers(db.get_collection())
    writer = Writer()
    features = Features()
    features.average_horest = np.zeros((1, 18))[0].tolist()
    features.average_texture = np.zeros((1, 18))[0].tolist()

    dummy = np.zeros((1, 18))[0].tolist()
    dummy_SDS = np.ones((1, 300))[0].tolist()
    dummy_SOH1 = np.ones((1, 200))[0].tolist()
    dummy_SOH2 = np.zeros((1, 500))[0].tolist()

    horest = []
    horest.append(dummy)
    horest.append(dummy)
    horest.append(dummy)
    horest.append(dummy)
    features.horest_features = horest

    texture = []
    texture.append(dummy)
    texture.append(dummy)
    texture.append(dummy)
    features.texture_feature = texture

    SDS = []
    SOH = []
    SDS.append(dummy_SDS)
    SDS.append(dummy_SDS)
    features.sift_SDS = SDS

    SOH.append(dummy_SOH1)
    SOH.append(dummy_SOH2)
    features.sift_SOH = SOH

    # writer.id = 1
    # writer.name = "Samar Gamal"

    # writer.id = 2
    # writer.name = "May Ahmed"

    # writer.id = 3
    # writer.name = "Omar Shaalan"

    writer.id = 4
    writer.name = "Ahmed Gamal"

    name_splitted = writer.name.split()
    writer.username = name_splitted[0][0].lower() + name_splitted[1].lower() + str(writer.id)
    writer.features = features
    status_code, message = writers.create_writer(writer)
    # status_code, message = writers.update_writer(writer)

    # writer = writers.get_writer(4)

    # writers_objs = writers.get_features([5])
    # for writer_obj in writers_objs:
    #     print(writer_obj.id)

    raise ExceptionHandler(message=message.value, status_code=status_code.value)


if __name__ == "__main__":
    app.run()


# # Create writer
# db = Database()
# # db.create_database()
# # db.create_collection()
# db.connect()
#
# writers = Writers(db.get_collection())
# writer = Writer()
# features = Features()
# features.average_horest = np.zeros((1, 18))[0].tolist()
# features.average_texture = np.zeros((1, 18))[0].tolist()
#
# dummy = np.zeros((1, 18))[0].tolist()
# dummy_SDS = np.ones((1, 300))[0].tolist()
# dummy_SOH1 = np.ones((1, 200))[0].tolist()
# dummy_SOH2 = np.zeros((1, 500))[0].tolist()
#
# horest = []
# horest.append(dummy)
# horest.append(dummy)
# horest.append(dummy)
# features.horest_features = horest
#
# texture = []
# texture.append(dummy)
# texture.append(dummy)
# texture.append(dummy)
# features.texture_feature = texture
#
# SDS = []
# SOH = []
# SDS.append(dummy_SDS)
# SDS.append(dummy_SDS)
# features.sift_SDS = SDS
#
# SOH.append(dummy_SOH1)
# SOH.append(dummy_SOH2)
# features.sift_SOH = SOH
#
# # writer.id = 1
# # writer.name = "Samar Gamal"
#
# # writer.id = 2
# # writer.name = "May Ahmed"
#
# writer.id = 3
# writer.name = "Omar Shaalan"
#
# name_splitted = writer.name.split()
# writer.username = name_splitted[0][0].lower() + name_splitted[1].lower() + str(writer.id)
# writer.features = features
# status_code = writers.create_writer(writer)
#
