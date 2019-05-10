class WritersVo:
    """
    Class for writers model, to return it to the application
    """
    def __init__(self):
        """
        Constructor for writers model
        """
        self._id = None
        self._name = None
        self._username = None

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._name

    @property
    def username(self):
        return self._username

    @id.setter
    def id(self, value):
        self._id = value

    @name.setter
    def name(self, value):
        self._name = value

    @username.setter
    def username(self, value):
        self._username = value