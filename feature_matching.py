import numpy as np


# u: SDS of I1 (first image)
# v: SDS of I2 (second image)
# x: SOH of I1 (first image)
# y: SOH of I2 (second image)
def match(u, v, x, y, w):

    # Manhattan distance to measure the dissimilarity between two SDSs u and v
    D1 = np.sum(np.abs(u - v))

    # Chi-Square distance to measurethe dissimilarity between SOH x and y
    if(x.shape[1] != y.shape[1]):
        if(x.shape[1] < y.shape[1]):
            padding = np.zeros((x.shape[0], (y.shape[1] - x.shape[1])))
            x = np.append(x, padding,axis=1)
        else:
            padding = np.zeros((y.shape[0], (x.shape[1] - y.shape[1])))
            y = np.append(y, padding,axis=1)


    D2 = np.sum(np.square(x - y) / (x + y + 1e-16))

    # normalize two distance between [0, 1]
    distances = [D1, D2]
    distances = distances / np.max(distances)

    # D new distance to measure the dissimilarity between I1 and I2
    D = w * distances[0] + (1 - w) * distances[1]

    return D

