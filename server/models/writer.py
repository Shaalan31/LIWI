from server.models.features import *


class Writer:
    """
    Class for writer model contains:
    1. writer id
    2. writer name
    3. writer username
    4. writer features
    """
    def __init__(self):
        """
        Class constructor
        """
        self._id = None
        self._name = None
        self._username = None
        self._address = None
        self._phone = None
        self._nid = None
        self._image = None
        self._birthday = None
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
    def address(self):
        return self._address

    @property
    def phone(self):
        return self._phone

    @property
    def nid(self):
        return self._nid

    @property
    def image(self):
        return self._image

    @property
    def birthday(self):
        return self._birthday

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

    @address.setter
    def address(self, value):
        self._address = value

    @phone.setter
    def phone(self, value):
        self._phone = value

    @nid.setter
    def nid(self, value):
        self._nid = value

    @image.setter
    def image(self, value):
        self._image = value

    @birthday.setter
    def birthday(self, value):
        self._birthday = value

    @features.setter
    def features(self, value):
        self._features = value
