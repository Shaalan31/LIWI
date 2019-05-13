class WriterVo:
    def __init__(self, id, name, username, image, address, phone, birthday, nid):
        self._id = id
        self._name = name
        self._username = username
        self._image = image
        self._address = address
        self._phone = phone
        self._birthday = birthday
        self._nid = nid

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._name

    @property
    def username(self):
        return self._username
