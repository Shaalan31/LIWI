from server.utils.utilities import *


class Writers:
    """
    Class of writers dao
    """
    def __init__(self, collection):
        """
        Constructor for writers dao
        :param collection: collection of database
        """
        self.collection = collection

    def create_writer(self, writer):
        """
        Create a new writer in the collection
        :param writer: writer model
        :return: HttpErrors, HttpMessages
        """
        writer_exists = self.collection.find({"_username": writer.username})
        if writer_exists.count() == 0:
            writer_dict = writer_to_dict(writer)
            self.collection.insert_one(writer_dict)

            return HttpErrors.SUCCESS, HttpMessages.CREATE_SUCCESS
        else:
            return HttpErrors.CONFLICT, HttpMessages.CONFLICT

    def update_writer(self, writer):
        """
        Update a writer in the collection
        :param writer: writer model
        :return: HttpErrors, HttpMessages
        """
        writer_exists = self.collection.find({"_id": writer.id, "_username": writer.username})
        if writer_exists.count() == 0:
            return HttpErrors.NOTFOUND, HttpMessages.NOTFOUND
        else:
            writer_dict = writer_to_dict(writer)
            self.collection.update({"_id": writer.id, "_username": writer.username}, {"$set": writer_dict})

            return HttpErrors.SUCCESS, HttpMessages.UPDATE_SUCCESS

    def get_writer(self, writer_id):
        """
        Get writer from the collection
        :param writer_id: int
        :return: writer model if it exists, if it does not return none
        """
        writer = self.collection.find({"_id": writer_id})
        if writer.count() == 1:
            writer_obj = dict_to_writer(writer[0])
            return writer_obj
        else: return None

    def get_features(self, writers_ids):
        """
        Get all features with the writers' name, usernames and ids
        :param writers_ids:
        :return: list of writer model
        """
        writers = []
        writers_dicts = self.collection.find({"_id": {"$in": writers_ids}})
        if writers_dicts.count() != 0:
            for writer_dict in writers_dicts:
                writer = dict_to_writer(writer_dict)
                writers.append(writer)
        return writers

    def get_writers_not_none(self):
        """
        Get writers' ids, names and usernames for the application
        :return: list of writers
        """
        writers = []
        writers_dicts = self.collection.find({"_features._sift_SDS": { "$exists": True, "$ne": []},
                                              "_features._sift_SOH": { "$exists": True, "$ne": []}})
        if writers_dicts.count() != 0:
            for writer_dict in writers_dicts:
                writer = dict_to_writers(writer_dict)
                writers.append(writer)
            return HttpErrors.SUCCESS, HttpMessages.SUCCESS, writers
        else:
            return HttpErrors.NOTFOUND, HttpMessages.NOWRITERS, None

    def get_writers(self):
        """
        Get writers' ids, names and usernames for the application
        :return: list of writers
        """
        writers = []
        writers_dicts = self.collection.find()
        if writers_dicts.count() != 0:
            for writer_dict in writers_dicts:
                writer = dict_to_writers(writer_dict)
                writers.append(writer)
            return HttpErrors.SUCCESS, HttpMessages.SUCCESS, writers
        else:
            return HttpErrors.NOTFOUND, HttpMessages.NOWRITERS, None

    def get_writers_count(self):
        """
        Get All Writers count
        :return: Writers Count
        """
        return self.collection.count()

    def get_writer_profile(self, writer_id, host_url):
        """
        Get Writer's Profile from database
        :param writer_id: writer id
        :return: HttpErrors:
                - 200 for success
                - 404 for not found
                HttpMessages:
                - "OK" for success
                - "Writer not found" if writer is not found
                ProfileVo Object or None if writer is not found
        """
        writer = self.collection.find({"_id": int(writer_id)}, {"_id": 1, "_name": 1, "_username": 1, "_address": 1, "_phone": 1, "_nid": 1, "_image": 1, "_birthday": 1})
        if writer.count() == 1:
            profile_obj = dict_to_profile(writer[0], host_url)
            return HttpErrors.SUCCESS, HttpMessages.SUCCESS, profile_obj
        else:
            return HttpErrors.NOTFOUND, HttpMessages.NOTFOUND, None

