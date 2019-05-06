from enum import Enum


class HttpMessages(Enum):
    SUCCESS = "Writer Created Successfully"
    NOTFOUND = "Writer is not found"
    CONFLICT = "Writer already exists"