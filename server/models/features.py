import numpy as np
class Features:
    def __init__(self):
        self._horest_features = []
        self._average_horest = []
        self._texture_feature = []
        self._average_texture = []
        self._sift_SDS = []
        self._sift_SOH = []

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

    @horest_features.setter
    def horest_features(self, value):
        self._horest_features = value

    @average_horest.setter
    def average_horest(self, value):
        self._average_horest = value

    @texture_feature.setter
    def texture_feature(self, value):
        self._texture_feature = value

    @average_texture.setter
    def average_texture(self, value):
        self._average_texture = value

    @sift_SDS.setter
    def sift_SDS(self, value):
        self._sift_SDS = value

    @sift_SOH.setter
    def sift_SOH(self, value):
        self._sift_SOH = value