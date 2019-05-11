from enum import Enum


class HttpErrors(Enum):
    SUCCESS = 200
    NOTFOUND = 404
    CONFLICT = 409
    BADREQUEST = 400
