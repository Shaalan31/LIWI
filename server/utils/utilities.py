from server.models.writer import *
from server.views.writersvo import *
from server.views.profilevo import *
from server.models.features import *
from server.httpresponses.errors import *
from server.httpresponses.messages import *
import re


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
    features.sift_SDS = features_dict["_sift_SDS"]
    features.sift_SOH = features_dict["_sift_SOH"]

    writer.features = features

    return writer


def dict_to_profile(writer_dict):
    """
    Convert writer dictionary into ProfileVo object
    :param writer_dict: dictionary contains writer's attributes
    :return: ProfileVo object
    """
    profile = ProfileVo()
    profile.id = writer_dict["_id"]
    profile.name = writer_dict["_name"]
    profile.username = writer_dict["_username"]
    profile.address = writer_dict["_address"]
    profile.phone = writer_dict["_phone"]
    profile.nid = writer_dict["_nid"]
    profile.image = writer_dict["_image"]

    return profile


def dict_to_writers(writer_dict):
    """
    Convert writer dictionary into writers model
    :param writer_dict: dictionary returned from database
    :return: writer object of writers model
    """
    writer = WritersVo()
    writer.id = writer_dict["_id"]
    writer.name = writer_dict["_name"]
    writer.username = writer_dict["_username"]

    return writer


def validate_writer_request(request):
    """
    Validate the create writer request regarding phone format, and national ID format
    :param request:
    :return: HTTP Error code:
                - 200 for success
                - 400 if validation failed
             HTTP Message:
                - "OK" for success
                - "Phone is invalid" for invalid phone format
                - "National ID is invalid" for invalid National ID format
    """
    phone_pattern = re.compile("(01)[0 1 2 5][0-9]{8}")
    match_phone = phone_pattern.match(request["_phone"])

    nid_pattern = re.compile("(2|3)[0-9][1-9][0-1][1-9][0-3][1-9](01|02|03|04|11|12|13|14|15|16|17|18|19|21|22|23|24|25|26|27|28|29|31|32|33|34|35|88)\d\d\d\d\d")
    match_nid = nid_pattern.match(request["_nid"])

    if match_phone is None:
        return HttpErrors.BADREQUEST, HttpMessages.INVALIDPHONE
    elif match_nid is None:
        return HttpErrors.BADREQUEST, HttpMessages.INVALIDNID
    else:
        return HttpErrors.SUCCESS, HttpMessages.SUCCESS


def func(writer):
    """
    Function to return attribute writer id
    :param writer: writer model object
    :return: attribute id
    """
    return writer.id
