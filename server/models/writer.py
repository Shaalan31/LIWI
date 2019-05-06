from server.models.features import *


class Writer:
    def __init__(self):
        self._id = None
        self._name = None
        self._username = None
        self._features = Features()

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._name

    @property
    def username(self):
        return self._username

    @property
    def features(self):
        return self._features

    @id.setter
    def id(self, value):
        self._id = value

    @name.setter
    def name(self, value):
        self._name = value

    @username.setter
    def username(self, value):
        self._username = value

    @features.setter
    def features(self, value):
        self._features = value