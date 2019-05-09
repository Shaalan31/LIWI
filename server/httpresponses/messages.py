from enum import Enum


class HttpMessages(Enum):
    SUCCESS = "OK"
    CREATE_SUCCESS = "Writer Created Successfully"
    UPDATE_SUCCESS = "Writer Updated Successfully"
    NOTFOUND = "Writer is not found"
    CONFLICT = "Writer already exists"
    CONFLICT_PREDICTION = "Error in prediction"
    NOWRITERS = "No writers found"