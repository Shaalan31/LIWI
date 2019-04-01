import cv2
import skimage.io as io
import numpy as np
from skimage.filters import threshold_otsu
from skimage.filters import gaussian
from preprocessing import *

def word_segmentation(image):

    image_copy = remove_shadow(image.copy())

    image_gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)

    # convert into binary image using Otsu
    image_binary = image_gray * 255
    threshold = threshold_otsu(image_binary)
    image_binary[(image_binary > threshold)] = 255
    image_binary[(image_binary <= threshold)] = 0

    cv2.imwrite('image_otsu.png', image_binary)

    # extract handwriting from image
    # top, bottom = extract_text(image_binary)
    # image_binary = image_binary[top:bottom, :]

    # get all connected components
    im, contours, hierarchy = cv2.findContours(np.copy(image_binary), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_rect = np.zeros((len(contours), 1))

    for i in range(0, len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        bounding_rect[i] = (int(h))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imwrite('image_contours.png', image)

    # get the average height ha of all CCs in Ib to decide the variance
    variance = np.average(bounding_rect[:, 0]) / 5
    image_gaussian = gaussian(image_binary.copy(), sigma=variance) * 255
    cv2.imwrite('image_gaussian.png', image_gaussian)

    # convert gaussian image into binary image using Otsu
    image_gaussian_binary = image_gaussian.copy().astype('uint8')
    threshold = threshold_otsu(image_gaussian_binary)
    image_gaussian_binary[(image_gaussian_binary > threshold)] = 255
    image_gaussian_binary[(image_gaussian_binary <= threshold)] = 0
    cv2.imwrite('image_gaussian_otsu.png', image_gaussian_binary)

    # get contours from binarized gaussian image
    bounding_rects = np.zeros((len(contours), 4))
    im, contours, hierarchy = cv2.findContours(np.copy(image_gaussian_binary), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(0, len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        bounding_rects[i] = (int(x), int(y), int(w), int(h))
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imwrite('image_final_contours.png', image_copy)

image = cv2.imread('sample1.png')
word_segmentation(image)