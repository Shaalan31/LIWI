from flask.json import JSONEncoder
import numpy as np


class WriterEncoder(JSONEncoder):
    """
    Class to encode writer object into json
    """
    def default(self, o):
        """
        Encode object into json
        :param o: object to be converted into json
        :return: object dictionary
        """
        if isinstance(o, np.ndarray):
            return o.tolist()

        return o.__dict__
