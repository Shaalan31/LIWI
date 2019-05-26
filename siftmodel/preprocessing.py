import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage.filters import gaussian
from utils.filters import *


class Preprocessing:
    def _init(self):
        pass

    # remove shadow from the image
    @staticmethod
    def remove_shadow(img):
        dilated = cv2.dilate(img, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated, 21)
        diff_img = 255 - cv2.absdiff(img, bg_img)
        norm_img = diff_img.copy()
        cv2.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        _, th_img = cv2.threshold(norm_img, 230, 0, cv2.THRESH_TRUNC)
        cv2.normalize(th_img, th_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        return th_img

    # extract text from IAM database
    @staticmethod
    def extract_text(img):
        filters = Filters()

        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Noise removal with gaussian
        # img = filters.gaussian(img, 1)
        img = gaussian(img, 1)

        img = img * 255
        threshold = threshold_otsu(img)
        # threshold = filters.threshold_otsu(img)
        img[(img > threshold)] = 255
        img[(img <= threshold)] = 0

        horizontal = np.copy(img)
        cols = horizontal.shape[1]
        horizontal_size = int(cols / 15)
        horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
        horizontal = cv2.dilate(horizontal, horizontalStructure)
        horizontal = cv2.erode(horizontal, horizontalStructure)
        horizontal = 255 - horizontal
        horizontal = horizontal / 255
        sum = np.sum(horizontal, axis=1)
        sum[sum < int(cols / 6)] = 0
        sum[sum > int(cols / 6)] = 1
        count_lines = len(np.argwhere(np.diff(np.argwhere(sum == 1), axis=0) > 2)) + 1
        if np.max(sum) == np.min(sum) or count_lines < 3:
            return 0, img.shape[0]
        half = int(sum.shape[0] / 2)
        top_boundary = half - np.argmax(sum[half:0:-1])
        bottom_boundary = half + np.argmax(sum[half:])

        return top_boundary + 2, bottom_boundary - 2