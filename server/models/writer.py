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
