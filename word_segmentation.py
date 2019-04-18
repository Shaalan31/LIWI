import cv2
import math
from skimage.filters import threshold_otsu
from skimage.filters import gaussian
from preprocessing import *

def within_word_distance(bounding_rect_sorted):
    diff_dist_word = np.diff(bounding_rect_sorted, axis=0)[:, 0] - bounding_rect_sorted[:-1, 2]
    threshold = np.average(np.abs(diff_dist_word))

    within_word_dist = np.average(np.abs(diff_dist_word[np.where(diff_dist_word < threshold)]))
    if math.isnan(within_word_dist):
        within_word_dist = 0

    return diff_dist_word, within_word_dist

def word_distance(bounding_rect_sorted):
    diff_dist_word = np.diff(bounding_rect_sorted, axis=0)[:, 0] - bounding_rect_sorted[:-1, 2]
    threshold = np.average(np.abs(diff_dist_word))

    word_dist = np.average(diff_dist_word[np.where(diff_dist_word > threshold)])
    # if line consists of only one word
    if math.isnan(word_dist):
        word_dist = 0

    return word_dist


def segment(image_gray):
    # Noise removal with gaussian
    image_gray = gaussian(image_gray, 1)

    # Thresholding
    image_gray *= 255
    threshold = np.round(threshold_otsu(image_gray) * 1.1)
    image_gray[(image_gray > threshold)] = 255
    image_gray[(image_gray <= threshold)] = 0

    top, bottom = extract_text(image_gray)
    imageGray = image_gray[top:bottom, :]

    indexes_lines = []
    line_start = 0
    found_line = False
    for line_index in range(imageGray.shape[0]):
        values, count = np.unique(imageGray[line_index, :], return_counts=True)
        if len(values) == 1:
            found_line = False
            continue
        countBlack = count[0]
        countWhite = count[1]
        total = countWhite + countBlack
        percentageBlack = (countBlack / total) * 100
        if percentageBlack > 1 and not found_line:
            found_line = True
            line_start = line_index
        else:
            if found_line and percentageBlack < 1:
                if line_index - line_start > (imageGray.shape[0] / 60):
                    indexes_lines.append([line_start, line_index])
                    #line = imageGray[line_start:line_index, :].astype('uint8')
                    #writer_lines.append(line)
                    #cv2.imwrite("line.png", line)
                found_line = False

    indexes_lines = np.asmatrix(indexes_lines)

    return indexes_lines


def merge_swrs(image, image_gray, bounding_rects):
    image_copy = image.copy()

    # sort bounding rectsangles on y then on x
    bounding_rects_sorted = bounding_rects[np.lexsort((bounding_rects[:, 0], bounding_rects[:, 1]))]

    # get all indexes of lines in the paper
    indexes_lines = segment(image_gray)

    for index_line in indexes_lines:
        # get line by line
        line = bounding_rects_sorted[(bounding_rects_sorted[:, 5] >= index_line[0, 0] )& (bounding_rects_sorted[:, 5] <= index_line[0, 1])]

        # sort bounding rectangles on x
        line = line[line[:, 0].argsort()]

        # for i in range(0, len(line)):
        #     x = int(line[i, 0])
        #     y = int(line[i, 1])
        #     w = int(line[i, 2])
        #     h = int(line[i, 3])
        #     cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #     cv2.imwrite('image_sorted_contours.png', image_copy)


def word_segmentation(image):

    image_orig = np.copy(image)

    image = remove_shadow(image)

    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Noise removal with gaussian
    imageGray = gaussian(imageGray, 1)

    # Thresholding
    imageGray *= 255
    threshold = np.round(threshold_otsu(imageGray) * 1.1)
    imageGray[(imageGray > threshold)] = 255
    imageGray[(imageGray <= threshold)] = 0

    cv2.imwrite('image_otsu.png', imageGray)


    # extract handwriting from image
    #top, bottom = extract_text(image_binary)
    # image_binary = image_binary[top:bottom, :]
    # cv2.imwrite('image_extract_text.png', image_binary)

    # get all connected components
    im, contours, hierarchy = cv2.findContours(np.copy(imageGray), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    im, contours, hierarchy = cv2.findContours(np.copy(image_gaussian_binary), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_rects = np.zeros((len(contours), 7))


    for i in range(0, len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        bounding_rects[i] = (int(x), int(y), int(w), int(h), int(x + 0.5 * w), int(y + 0.5 * h), int(w * h))
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # check area of contours
    iAmDbImageSize = 375 / 8780618
    mask = (bounding_rects[:, 6] > (iAmDbImageSize * (image_binary.shape[0] * image_binary.shape[1]))).astype('int')
    bounding_rect_sorted = bounding_rects[np.where(mask)]
    cv2.imwrite('image_final_contours.png', image_copy)

    # merging the SWRs to get the word regions (WRs)
    merge_swrs(image_orig.copy(), image_gray.copy(), bounding_rects)

image = cv2.imread('a01-000u.png')
word_segmentation(image)