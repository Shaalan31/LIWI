from warnings import warn
from skimage.exposure import histogram
import numpy as np
import cv2


class Filters:
    """
    Class for filters implementation:
    - Otsu threshold
    - Gaussian filter
    - Dilation
    - Erosion
    """
    def __init__(self):
        pass

    def otsu_segmentation(self, image):
        nbins = 256

        # Check if the image has been converted into gray scale
        if len(image.shape) > 2 and image.shape[-1] in (3, 4):
            warn("Otsu Threshold works on gray scale images only!")

        # get histogram of the image
        histog, gray_levels = histogram(image.ravel(), nbins)
        # histog = histog.astype(float)

        # calculate probabilities of classes
        wo = np.cumsum(histog)
        w1 = np.cumsum(histog[::-1])[::-1]

        # calculate the variance between classes
        muo = np.cumsum(histog * gray_levels) / wo
        mu1 = (np.cumsum((histog * gray_levels)[::-1]) / w1[::-1])[::-1]
        var = wo[:-1] * w1[1:] * (muo[:-1] - mu1[1:]) ** 2

        threshold = gray_levels[:-1][np.argmax(var)]
        return threshold