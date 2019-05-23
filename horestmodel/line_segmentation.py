from utils.common_functions import *
from skimage.filters import gaussian
from skimage.filters import threshold_otsu
from utils.filters import *


class LineSegmentation:
    def __init__(self, image):
        self.image = image

    def segment(self):
        filters = Filters()

        image = remove_shadow(self.image)

        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Noise removal with gaussian
        imageGray = gaussian(imageGray, 1)

        # Thresholding
        imageGray *= 255
        threshold = np.round(threshold_otsu(imageGray) * 1.1)
        # threshold = np.round(filters.otsu_segmentation(imageGray) * 1.1)
        imageGray[(imageGray > threshold)] = 255
        imageGray[(imageGray <= threshold)] = 0

        top, bottom = extract_text(imageGray)
        imageGray = imageGray[top:bottom, :]

        writer_lines = []
        line_start = 0
        foundALine = False
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
                    if line_index - line_start > (imageGray.shape[0] / 60):
                        line = cv2.copyMakeBorder(imageGray[line_start:line_index, :].astype('uint8'), 1, 1, 1, 1,
                                                  cv2.BORDER_CONSTANT, value=[255, 255, 255])
                        writer_lines.append(line)
                    foundALine = False
        return writer_lines
