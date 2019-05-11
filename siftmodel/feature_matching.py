import numpy as np


class FeatureMatching:
    def __init__(self):
        pass

    # u: SDS of I1 (first image)
    # v: SDS of I2 (second image)
    # x: SOH of I1 (first image)
    # y: SOH of I2 (second image)
    def calculate_distances(self, u, v, x, y):

        # Manhattan distance to measure the dissimilarity between two SDSs u and v
        D1 = np.sum(np.abs(u - v), axis=1)

        # Chi-Square distance to measurethe dissimilarity between SOH x and y
        if(x.shape[1] != y.shape[1]):
            if(x.shape[1] < y.shape[1]):
                padding = np.zeros((x.shape[0], (y.shape[1] - x.shape[1])))
                x = np.append(x, padding,axis=1)
            else:
                padding = np.zeros((y.shape[0], (x.shape[1] - y.shape[1])))
                y = np.append(y, padding,axis=1)

        D2 = np.sum(np.square(x - y) / (x + y + 1e-16), axis=1)

        return D1, D2

    def match(self, manhattan, chi_square, w):

        # normalize two distances
        manhattan = (manhattan - np.min(manhattan)) / (np.max(manhattan) - np.min(manhattan))
        chi_square = (chi_square - np.min(chi_square)) / (np.max(chi_square) - np.min(chi_square))

        # D new distance to measure the dissimilarity between I1 and I2
        D = w * manhattan + (1 - w) * chi_square

        return np.argmin(D)

    def match2(self, manhattan, chi_square, w):

        # normalize two distances
        manhattan = (manhattan - np.min(manhattan)) / (np.max(manhattan) - np.min(manhattan))
        chi_square = (chi_square - np.min(chi_square)) / (np.max(chi_square) - np.min(chi_square))

        # D new distance to measure the dissimilarity between I1 and I2
        D = w * manhattan + (1 - w) * chi_square
        result = np.array([np.argmin(D),0,0])
        return np.argmin(D)
