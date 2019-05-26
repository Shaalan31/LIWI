from warnings import warn
from skimage.exposure import histogram
from scipy.ndimage.filters import convolve
from scipy.signal import convolve2d
import numpy as np
import scipy
import math


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

    def threhold_otsu(self, image):
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

    def gausian(self, image, sigma):
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

    def sobelv(self, image):
        """
        Function to implement sobel_v filter
        :param image
        :return: image after applying filter
        """
        sobel_v = np.array([[-1,0,1], [-2,0,2], [-1,0,1]]) / 4.0

        sobel_img = convolve2d(image, sobel_v)

        return sobel_img

    def sobelh(self, image):
        """
        Function to implement sobel_h filter
        :param image
        :return: image after applying filter
        """
        sobel_h = np.array([[-1,-2,-1], [0,0,0], [1,2,1]]) / 4.0

        sobel_img = convolve2d(image, sobel_h)

        return sobel_img

    def erod(self, image, windowWidth, windowHeight):
        """
        Function to implement erosion
        :param image
        :param windowWidth
        :param windowHeight
        :return: image after applying filter
        """
        outputPixelValue = np.ones((image.shape[0], image.shape[1]))

        for x in range(0, image.shape[0] - 1):
            for y in range(0, image.shape[1] - 1):
                maskArray = np.ones((windowWidth, windowHeight))
                for fx in range(0, windowWidth):
                    for fy in range(0, windowHeight):
                        maskArray[fx][fy] = image[x + fx - 1][y + fy - 1]
                outputPixelValue[x][y] = np.min(maskArray)
        return outputPixelValue

    def dilat(self, image, windowWidth, windowHeight):
        """
        Function to implement dilation
        :param image
        :param windowWidth
        :param windowHeight
        :return: image after applying filter
        """
        outputPixelValue = np.ones((image.shape[0], image.shape[1]))

        for x in range(0, image.shape[0] - 1):
            for y in range(0, image.shape[1] - 1):
                maskArray = np.ones((windowWidth, windowHeight))
                for fx in range(0, windowWidth):
                    for fy in range(0, windowHeight):
                        maskArray[fx][fy] = image[x + fx - 1][y + fy - 1]
                outputPixelValue[x][y] = np.max(maskArray)
        return outputPixelValue

    def medianblur(self, image, windowWidth, windowHeight):
        edgex = math.floor(windowWidth / 2)
        edgey = math.floor(windowHeight / 2)
        outputPixelValue = np.ones((image.shape[0], image.shape[1]))

        for x in range(edgex, image.shape[0] - edgex):
            for y in range(edgey, image.shape[1] - edgey):
                colorArray = np.zeros((windowWidth, windowHeight))
                for fx in range(0, windowWidth):
                    for fy in range(0, windowHeight):
                        colorArray[fx][fy] = image[x + fx - edgex][y + fy - edgey]
                outputPixelValue[x][y] = np.median(colorArray)
        return outputPixelValue