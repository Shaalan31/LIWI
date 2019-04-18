import cv2
import numpy as np
from skimage.filters import gaussian
from skimage.filters import threshold_otsu

# remove shadow from the image
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
def extract_text(img):
    horizontal = np.copy(img)
    cols = horizontal.shape[1]
    horizontal_size = int(cols / 15)
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.dilate(horizontal, horizontalStructure)
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = 255 - horizontal
    horizontal = horizontal /255
    print(horizontal.shape)
    sum = np.sum(horizontal, axis=1)
    print(sum.shape)
    sum[sum < int(cols / 10)] = 0
    sum[sum > int(cols / 10)] = 1
    if np.max(sum) == np.min(sum):
        return 0, img.shape[0]
    print(sum.shape)
    half = int(sum.shape[0] / 2)
    print(half,'\n')
    print(sum)
    top_boundary = half - np.argmax(sum[half:0:-1])
    bottom_boundary = half + np.argmax(sum[half:])

    return top_boundary + 2, bottom_boundary - 2



image = cv2.imread('a01-000u.png')

image = remove_shadow(image)

imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Noise removal with gaussian
imageGray = gaussian(imageGray, 1)

# Thresholding
imageGray *= 255
threshold = np.round(threshold_otsu(imageGray) * 1.1)
imageGray[(imageGray > threshold)] = 255
imageGray[(imageGray <= threshold)] = 0

#print(extract_text(imageGray))