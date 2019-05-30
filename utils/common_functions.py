import matplotlib.pyplot as plt
import numpy as np
import cv2
from texturemodel.bounding_rect import *


# Show the figures / plots inside the notebook
def show_images(images, titles=None):
    # This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(1, n_ims, n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()


#
def showHist(imgHist):
    # An "interface" to matplotlib.axes.Axes.hist() method
    plt.figure()
    # imgHist = histogram(img, nbins=256)

    plt.bar(imgHist[1].astype(np.uint8), imgHist[0], width=0.8, align='center')


def remove_shadow(img):
    dilated = cv2.dilate(img, np.ones((7, 7), np.uint8))
    bg_img = cv2.medianBlur(dilated, 21)
    diff_img = 255 - cv2.absdiff(img, bg_img)
    norm_img = diff_img.copy()
    cv2.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    _, th_img = cv2.threshold(norm_img, 230, 0, cv2.THRESH_TRUNC)
    cv2.normalize(th_img, th_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return th_img


def extract_text(img):
    horizontal = np.copy(img)
    cols = horizontal.shape[1]
    horizontal_size = int(cols / 15)
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.dilate(horizontal, horizontalStructure)
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = 255 - horizontal
    horizontal /= 255
    # show_images([horizontal])
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


def getBoundingRects(image):
    image_shape = image.shape
    image = image.astype('uint8')

    small_components_ratio = 375 / 8780618

    all_bounding_rects = np.asarray([])
    _, contours, hierarchy = cv2.findContours(np.subtract(255, image.copy()), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = np.asarray(contours)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if int(w * h) > small_components_ratio * (image_shape[0] * image_shape[1]):
            # we need to discard this bounding rect since it's not logical
            if h > 256:
                continue
            new_bounding_rect = BoundingRect(h, w, np.divide(image[y:y + h, x:x + w], 255))
            all_bounding_rects = np.append(all_bounding_rects, new_bounding_rect)

    return all_bounding_rects


def feature_normalize(X):
    mean = np.mean(X, axis=0)
    normalized_X = X - mean
    deviation = np.sqrt(np.var(normalized_X, axis=0))
    normalized_X = np.divide(normalized_X, deviation)
    return normalized_X, mean, deviation

