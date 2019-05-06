from server.httpresponses.errors import *
from server.httpresponses.messages import *
from server.utils.utilities import *


class Writers:
    def __init__(self, collection):
        self.collection = collection

    def create_writer(self, writer):
        writer_exists = self.collection.find({"_id": writer.id, "_username": writer.username})
        if writer_exists.count() == 0:
            writer_dict = writer_to_dict(writer)
            self.collection.insert_one(writer_dict)

            return HttpErrors.SUCCESS, HttpMessages.SUCCESS
        else:
            return HttpErrors.CONFLICT, HttpMessages.CONFLICT

    def update_writer(self, writer):
        writer_exists = self.collection.find({"_id": writer.id, "_username": writer.username})
        if writer_exists.count() == 0:
            return HttpErrors.NOTFOUND, HttpMessages.NOTFOUND
        else:
            writer_dict = writer_to_dict(writer)
            self.collection.update({"_id": writer.id, "_username": writer.username}, {"$set": writer_dict})

            return HttpErrors.SUCCESS, HttpMessages.SUCCESS

    def get_writer(self, writer_id):
        return

    def get_features(self, writers):
        return

    def get_writers(self):
        return

