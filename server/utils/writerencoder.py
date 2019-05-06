from flask.json import JSONEncoder
import numpy as np


class WriterEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()

        return o.__dict__
