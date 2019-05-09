import pymongo


class Database:
    """
    Class for connecting to database
    """
    def __init__(self):
        """
        Constructor for database class
        """
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = None
        self.collection = None
        self.collection_arabic = None

    def create_database(self):
        """
        Create database for the first time only
        :return:
        """
        databases = self.client.list_database_names()
        print(databases)
        if "LIWI" not in databases:
            self.db = self.client["LIWI"]

    def create_collection(self):
        """
        Create writers collection for the first time only
        :return:
        """
        collections = self.db.list_collection_names()
        if "writers" not in collections:
            self.db.create_collection("writers")

        if "writers_arabic" not in collections:
            self.db.create_collection("writers_arabic")

    def connect(self):
        """
        Connect to database
        :return:
        """
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

        if "writers_arabic" in collections:
            self.collection_arabic = self.db["writers_arabic"]
        else:
            print('Collection Arabic not created')

    def get_client(self):
        """
        Returns database client
        :return:
        """
        return self.client

    def get_database(self):
        """
        Returns database
        :return:
        """
        return self.db

    def get_collection(self):
        """
        Returns collection
        :return:
        """
        return self.collection

    def get_collection_arabic(self):
        """
        Returns arabic collection
        :return:
        """
        return self.collection_arabic