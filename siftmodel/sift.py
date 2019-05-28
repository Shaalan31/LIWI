import cv2
import numpy as np


class Sift:

    def __init__(self):
        pass

    @staticmethod
    def unpack_sift_octave(kpt):
        _octave = kpt.octave
        octave = _octave & 0xFF
        layer = (_octave >> 8) & 0xFF
        if octave >= 128:
            octave |= -128
        if octave >= 0:
            scale = float(1 / (1 << octave))
        else:
            scale = float(1 << -octave)
        return (octave, layer, scale)

    def get_keypoints(self, img_gray):
        # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create(nOctaveLayers=3)
        kp, des = sift.detectAndCompute(img_gray, None)
        key_points = np.zeros((len(kp), 3))
        for i in range(len(kp)):
            octave, layer, scale = self.unpack_sift_octave(kp[i])
            # print(str(octave) + ', ' + str(layer) + ', ' + str(scale) + ', ' + str(kp[i].octave))
            key_points[i, 0] = kp[i].angle
            key_points[i, 1] = octave
            key_points[i, 2] = layer
        return key_points, des, kp

    def get_des(self, img_gray):
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(img_gray, None)
        return des
