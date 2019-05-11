import numpy as np
class Features:
    def __init__(self):
        self._horest_features = []
        self._texture_feature = []
        self._sift_SDS = []
        self._sift_SOH = []

    @property
    def horest_features(self):
        return self._horest_features

    @property
    def texture_feature(self):
        return self._texture_feature

    @property
    def sift_SDS(self):
        return self._sift_SDS

    @property
    def sift_SOH(self):
        return self._sift_SOH

    @horest_features.setter
    def horest_features(self, value):
        self._horest_features = value

    @texture_feature.setter
    def texture_feature(self, value):
        self._texture_feature = value

    @sift_SDS.setter
    def sift_SDS(self, value):
        self._sift_SDS = value

    @sift_SOH.setter
    def sift_SOH(self, value):
        self._sift_SOH = value