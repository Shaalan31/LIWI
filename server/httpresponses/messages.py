from enum import Enum


class HttpMessages(Enum):
    CREATE_SUCCESS = "Writer Created Successfully"
    UPDATE_SUCCESS = "Writer Updated Successfully"
    NOTFOUND = "Writer is not found"
    CONFLICT = "Writer already exists"