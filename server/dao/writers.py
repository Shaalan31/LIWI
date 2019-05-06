from server.httpresponses.errors import *
from server.httpresponses.messages import *
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
        writer_exists = self.collection.find({"_id": writer.id, "_username": writer.username})
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
        :return: writer model
        """
        writer = self.collection.find({"_id": writer_id})
        if writer.count() == 1:
            writer_obj = dict_to_writer(writer[0])

        return writer_obj

    def get_features(self, writers_ids):
        """
        Get all features with the writers' name, usernames and ids
        :param writers_ids:
        :return: list of writer model
        """
        return

    def get_writers(self):
        """
        Get writers' ids, names and usernames for the application
        :return: list of writers
        """
        return

