from server.models.writer import *
from server.models.features import *

def writer_to_dict(writer):
    """
    Convert writer object into dictionary
    :param writer: writer model
    :return: writer_dict: dictionary containing writer's info
    """
    writer_features_dict = writer.features.__dict__
    features = {'_features': writer_features_dict}
    writer_dict = writer.__dict__
    writer_dict.update(features)

    return writer_dict


def dict_to_writer(writer_dict):
    """
    Convert dictionary to writer object
    :param writer_dict:
    :return: writer model (object)
    """
    writer = Writer()
    writer.id = writer_dict["_id"]
    writer.name = writer_dict["_name"]
    writer.username = writer_dict["_username"]

    features_dict = writer_dict["_features"]
    features = Features()
    features.horest_features = features_dict["_horest_features"]
    features.texture_feature = features_dict["_texture_feature"]
    features.average_horest = features_dict["_average_horest"]
    features.average_texture = features_dict["_average_texture"]
    features.sift_SDS = features_dict["_sift_SDS"]
    features.sift_SOH = features_dict["_sift_SOH"]

    writer.features = features

    return writer