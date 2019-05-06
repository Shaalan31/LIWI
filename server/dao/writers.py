from server.httpresponses.errors import *
from server.httpresponses.messages import *


class Writers:
    def __init__(self, collection):
        self.collection = collection

    def create_writer(self, writer):
        writer_exists = self.collection.find({"_id": writer.id, "_username": writer.username})
        if writer_exists.count() == 0:
            writer_features_dict = writer.features.__dict__
            features = {'_features': writer_features_dict}
            writer_dict = writer.__dict__
            writer_dict.update(features)

            self.collection.insert_one(writer_dict)

            return HttpErrors.SUCCESS, HttpMessages.SUCCESS
        else:
            return HttpErrors.CONFLICT, HttpMessages.CONFLICT

    def update_writer(self, writer):
        return

    def get_writer(self, writer_id):
        return

    def get_features(self, writers):
        return

    def get_writers(self):
        return

