
class ExceptionHandler(Exception):
    """
    Class for handling exceptions
    """
    def __init__(self, message, status_code=None, data=None, payload=None):
        """
        Constructor for exception handler
        :param message: string
        :param status_code: int
        :param data: object
        :param payload:
        """
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload
        self.data = data

    def to_dict(self):
        """
        Convert response to dictionary
        :return: rv response as dictionary
        """
        rv = dict(self.payload or ())
        rv['message'] = self.message
        rv['status_code'] = self.status_code
        rv['data'] = self.data

        return rv