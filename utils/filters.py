from warnings import warn
from skimage.exposure import histogram
from scipy.ndimage.filters import convolve
import numpy as np
import scipy


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

    def threshold_otsu(self, image):
        """
        Function to implement Otsu Thresholding Algorithm
        :param image
        :return: Otsu Threshold
        """
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

    def gaussian(self, image, sigma):
        """
        Function to implement gaussian filter
        :param image
        :param sigma: parameter in gaussian
        :return: image after applying gaussian filter
        """
        kernel_width = 9
        half_width = int(kernel_width / 2)
        x, y = scipy.mgrid[-half_width:half_width + 1, -half_width:half_width + 1]

        gaussian_filter = np.exp((-(x ** 2 + y ** 2)) / (2 * sigma ** 2))
        gaussian = (gaussian_filter / np.sum(gaussian_filter)).astype(np.float32)

        gaussian_image = convolve(image, gaussian)

        return gaussian_image / 255

