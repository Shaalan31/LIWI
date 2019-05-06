import pymongo


class Database:
    def __init__(self):
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = None
        self.collection = None

    def create_database(self):
        databases = self.client.list_database_names()
        print(databases)
        if "LIWI" not in databases:
            self.db = self.client["LIWI"]

    def create_collection(self):
        collections = self.db.list_collection_names()
        if "writers" not in collections:
            self.db.create_collection("writers")

    def connect(self):
        # get database
        databases = self.client.list_database_names()
        if "LIWI" in databases:
            self.db = self.client["LIWI"]
        else:
            print('Database not created')

        # get collection
        collections = self.db.list_collection_names()
        if "writers" in collections:
            self.collection = self.db["writers"]
        else:
            print('Collection not created')

    def get_client(self):
        return self.client

    def get_database(self):
        return self.db

    def get_collection(self):
        return self.collection