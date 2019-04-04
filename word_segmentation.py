import cv2
from skimage.filters import threshold_otsu
from skimage.filters import gaussian
from preprocessing import *

def merge_swrs(image, bounding_rects):
    image_copy = image.copy()

    # sort bounding rectsangles on y then on x
    bounding_rects_sorted = bounding_rects[np.lexsort((bounding_rects[:, 0], bounding_rects[:, 1]))]

    # get difference between all ys
    diff_dist_lines = np.diff(bounding_rects_sorted, axis=0)[:, 1]

    # sort the difference between ys descending to get maximum differences (peaks / number of lines)
    diff_dist_lines_sorted = np.sort(diff_dist_lines)[::-1]

    # get the difference between the sorted array of differences between ys
    # diff_diff_dist_lines_sorted = np.diff(diff_dist_lines_sorted, axis=0)

    # get the index of the maximum value in the array diff_diff_dist_lines_sorted to get the index of the last peak
    # last_peak = np.argmin(diff_diff_dist_lines_sorted, axis=0)

    # get all peaks (blank lines)
    # peaks = diff_dist_lines_sorted[:last_peak+1]

    # maximum 15 lines
    threshold = np.average(diff_dist_lines_sorted[:15])

    peaks = diff_dist_lines_sorted[np.where(diff_dist_lines_sorted > threshold)]

    # get the indexes of blank lines
    mask = np.isin(diff_dist_lines, peaks)
    indexes = np.argwhere(mask == True)

    # lines splitted
    indexes = indexes.flatten() + 1
    lines = np.split(bounding_rects_sorted, indexes)

    # for line in lines:
    #     line = line[line[:, 0].argsort()]
    #     for i in range(0, len(line)):
    #         x = int(line[i, 0])
    #         y = int(line[i, 1])
    #         w = int(line[i, 2])
    #         h = int(line[i, 3])
    #         cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #         cv2.imwrite('image_sorted_contours.png', image_copy)


def word_segmentation(image):

    image_orig = image.copy()

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
    # cv2.imwrite('image_extract_text.png', image_binary)

    # get all connected components
    im, contours, hierarchy = cv2.findContours(np.copy(image_binary), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_rect = np.zeros((len(contours), 1))

    for i in range(0, len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        bounding_rect[i] = (int(h))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imwrite('image_contours.png', image)

    # get the average height ha of all CCs in Ib to decide the variance
    variance = np.average(bounding_rect[:, 0]) / 2.5
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
    bounding_rects = np.zeros((len(contours), 4))


    for i in range(0, len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        bounding_rects[i] = (int(x), int(y), int(w), int(h))
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imwrite('image_final_contours.png', image_copy)

    # merging the SWRs to get the word regions (WRs)
    merge_swrs(image_orig.copy(), bounding_rects)

image = cv2.imread('samplecrop3.png')
word_segmentation(image)