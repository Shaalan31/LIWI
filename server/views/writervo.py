class WriterVo:
    def __init__(self, id, name,username):
        self._id = id
        self._name = name
        self._username = username

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._name

    @property
    def username(self):
        return self._username
