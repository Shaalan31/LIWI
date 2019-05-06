# import json
from flask import json
from server.utils.writerencoder import *

class ExceptionHandler(Exception):

    def __init__(self, message, status_code=None, data=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload
        self.data = data

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        rv['status_code'] = self.status_code
        rv['data'] = self.data

        return rv