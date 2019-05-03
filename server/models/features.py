import numpy as np
class Features:
    def __init__(self):
        self._horest_features = np.asarray([])
        self._average_horest = np.asarray([])
        self._texture_feature = np.asarray([])
        self._average_texture = np.asarray([])
        self._sift_SDS = np.asarray([])
        self._sift_SOH = np.asarray([])

    @property
    def horest_features(self):
        return self._horest_features

    @property
    def average_horest(self):
        return self._average_horest

    @property
    def texture_feature(self):
        return self._texture_feature

    @property
    def average_texture(self):
        return self._average_texture

    @property
    def sift_SDS(self):
        return self._sift_SDS

    @property
    def sift_SOH(self):
        return self._sift_SOH
