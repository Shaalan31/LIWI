import numpy as np


def feature_normalize(X):
    mean = np.mean(X, axis=0)
    normalized_X = X - mean
    deviation = np.sqrt(np.var(normalized_X, axis=0))
    normalized_X = np.divide(normalized_X, deviation)
    return normalized_X, mean, deviation

