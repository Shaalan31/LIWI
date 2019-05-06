from enum import Enum


class HttpMessages(Enum):
    SUCCESS = "OK"
    CREATE_SUCCESS = "Writer Created Successfully"
    UPDATE_SUCCESS = "Writer Updated Successfully"
    NOTFOUND = "Writer is not found"
    CONFLICT = "Writer already exists"
    NOWRITERS = "No writers found"