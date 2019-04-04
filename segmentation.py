from commonfunctions import *
from skimage.filters import gaussian
from skimage.filters import threshold_otsu


def segment(image):
    image = remove_shadow(image)

    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Noise removal with gaussian
    imageGray = gaussian(imageGray, 1)

    # Thresholding
    imageGray *= 255
    threshold = np.round(threshold_otsu(imageGray) * 1.1)
    imageGray[(imageGray > threshold)] = 255
    imageGray[(imageGray <= threshold)] = 0

    # imageGray = cv2.erode(imageGray.copy(), np.ones((3, 3)), iterations=1)

    # show_images([imageGray])
    top, bottom = extract_text(imageGray)
    imageGray = imageGray[top:bottom, :]
    # show_images([imageGray])

    writer_lines = []
    line_start = 0
    foundALine = False
    # imgName = 0
    for line_index in range(imageGray.shape[0]):
        values, count = np.unique(imageGray[line_index, :], return_counts=True)
        if len(values) == 1:
            foundALine = False
            continue
        countBlack = count[0]
        countWhite = count[1]
        total = countWhite + countBlack
        percentageBlack = (countBlack / total) * 100
        if percentageBlack > 1 and not foundALine:
            foundALine = True
            line_start = line_index
        else:
            if foundALine and percentageBlack < 1:
                if line_index - line_start > (imageGray.shape[1] / 60):
                    line = cv2.copyMakeBorder(imageGray[line_start:line_index, :].astype('uint8'), 1, 1, 1, 1,
                                              cv2.BORDER_CONSTANT, value=[255, 255, 255])
                    # io.imsave('output/image' + str(imgName) + '.png', line)
                    # imgName += 1
                    # show_images([line])
                    writer_lines.append(line)
                foundALine = False
    return writer_lines


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
    sum[sum < int(cols / 10)] = 0
    sum[sum > int(cols / 10)] = 1
    if np.max(sum) == np.min(sum):
        return 0, img.shape[0]
    half = int(sum.shape[0] / 2)
    top_boundary = half - np.argmax(sum[half:0:-1])
    bottom_boundary = half + np.argmax(sum[half:])

    return top_boundary + 2, bottom_boundary-2
