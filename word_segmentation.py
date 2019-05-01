import cv2
from sift import *
from preprocessing import *


def segment(image_gray):
    # Noise removal with gaussian
    image_gray = gaussian(image_gray, 1)

    # Thresholding
    image_gray *= 255
    threshold = np.round(threshold_otsu(image_gray) * 1.1)
    image_gray[(image_gray > threshold)] = 255
    image_gray[(image_gray <= threshold)] = 0

    indexes_lines = []
    line_start = 0
    found_line = False
    for line_index in range(image_gray.shape[0]):
        values, count = np.unique(image_gray[line_index, :], return_counts=True)
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
                if line_index - line_start > (image_gray.shape[0] / 60):
                    indexes_lines.append([line_start, line_index])
                    #line = imageGray[line_start:line_index, :].astype('uint8')
                    #writer_lines.append(line)
                    #cv2.imwrite("line.png", line)
                found_line = False

    indexes_lines = np.asmatrix(indexes_lines)

    return indexes_lines


def merge_swrs(image_gray, bounding_rects, name):
    
    # sort bounding rectsangles on y then on x
    bounding_rects_sorted = bounding_rects[np.lexsort((bounding_rects[:, 0], bounding_rects[:, 1]))]

    # get all indexes of lines in the paper
    indexes_lines = segment(image_gray)

    sd = {}
    so = np.zeros((1,3))
    number = 0
    for index_line in indexes_lines:
        # get line by line
        line = bounding_rects_sorted[(np.bitwise_or (bounding_rects_sorted[:, 5] >= index_line[0, 0] ,
                                      bounding_rects_sorted[:, 1] >= index_line[0, 0] ))&
                                     (np.bitwise_or(bounding_rects_sorted[:, 5] <= index_line[0, 1] ,
                                      bounding_rects_sorted[:, 1] <= index_line[0, 1]))]

        # sort bounding rectangles on x
        line = line[line[:, 0].argsort()]

        # calculate distances between two words
        diff_dist_word = np.diff(line, axis=0)[:, 0] - line[:-1, 2]

        # get indexes of differences which are less than threshold
        diff_indexes = np.argwhere(diff_dist_word < 20)

        word_index=0
        while word_index < len(line):
            number += 1
            if word_index in diff_indexes:
                start = word_index
                while True:
                    word_index += 1
                    if(word_index not  in diff_indexes):
                        break

                ymax = int(np.max(line[start:word_index+1, 1]+line[start:word_index+1, 3]))
                ymin = int(np.min(line[start:word_index+1, 1]))

                # get segmented word from the image
                word = image_gray[ymin:ymax, int(line[start,0]):int(line[word_index,0]+line[word_index,2])]

                cv2.imwrite('words/' + str(int(number)) + '_' + str(name.replace('.png', '')) + '.png', word)

            else:
                # get segmented word from the image
                word = image_gray[int(line[word_index,1]):int(line[word_index,1]+line[word_index,3]),int(line[word_index,0]):int(line[word_index,0]+line[word_index,2])]

                cv2.imwrite('words/' + str(int(number)) + '_' + str(name.replace('.png', '')) + '.png', word)

            word_index += 1

            if (len(word) == 0):
                continue

            # get sift descriptors and orientation
            key_points, des = getKeypoints(word)
            sd.update({ number: des })
            so = np.append(so, key_points, axis=0)

    so = np.delete(so,0,0)
    return sd, so


def word_segmentation(image, name):

    image_orig = image.copy()

    image_copy = image_orig.copy()

    image_gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)

    # convert into binary image using Otsu
    image_binary = image_gray * 255
    threshold = threshold_otsu(image_binary)
    image_binary[(image_binary > threshold)] = 255
    image_binary[(image_binary <= threshold)] = 0
    #cv2.imwrite('image_otsu.png', image_binary)

    # get all connected components
    im, contours, hierarchy = cv2.findContours(np.copy(image_binary), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_rect = np.zeros((len(contours), 1))

    for i in range(0, len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        bounding_rect[i] = (int(h))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #cv2.imwrite('image_contours.png', image)

    # get the average height ha of all CCs in Ib to decide the variance
    variance = np.average(bounding_rect[:, 0]) / 5
    image_gaussian = gaussian(image_binary.copy(), sigma=variance) * 255
    #cv2.imwrite('image_gaussian.png', image_gaussian)

    # convert gaussian image into binary image using Otsu
    image_gaussian_binary = image_gaussian.copy().astype('uint8')
    threshold = threshold_otsu(image_gaussian_binary)
    image_gaussian_binary[(image_gaussian_binary > threshold)] = 255
    image_gaussian_binary[(image_gaussian_binary <= threshold)] = 0
    #cv2.imwrite('image_gaussian_otsu.png', image_gaussian_binary)

    # get contours from binarized gaussian image
    im, contours, hierarchy = cv2.findContours(np.copy(image_gaussian_binary), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_rects = np.zeros((len(contours), 6))

    # check area of contours
    iAmDbImageSize = 1600 / 8780618
    for i in range(0, len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        if(int(w*h) > (iAmDbImageSize * (image_orig.shape[0] * image_orig.shape[1]))):
            bounding_rects[i] = (int(x), int(y), int(w), int(h), int(x + 0.5 * w), int(y + 0.5 * h))
            cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #cv2.imwrite('image_final_contours.png', image_copy)

    # merging the SWRs to get the word regions (WRs)
    sd, so = merge_swrs(image_gray.copy(), bounding_rects, name)

    return sd, so