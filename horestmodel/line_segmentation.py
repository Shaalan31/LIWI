import errno
import os
import time

from skimage.filters import gaussian
from skimage.filters import threshold_otsu

from utils.common_functions import *
from utils.filters import *


class LineSegmentation:
    def __init__(self, image, socket=None):
        self.image = image
        self.socketIO = socket

    def segment(self):
        print("segment")
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

        # Khairy (sending thresholded image by otsu)
        if self.socketIO is not None:
            self.makeTempDirectory()
            file_name = 'thresholdedImage'
            file_name = self.saveImage(file_name, imageGray)

            # with self.thread_lock:
            #     self.thread = self.socketIO.start_background_task(self.sendData(file_name))
            self.socketIO.start_background_task(self.sendData(file_name, 'Thresholded'))

        top, bottom = extract_text(imageGray)
        imageGray = imageGray[top:bottom, :]

        countLines = 0
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
                        countLines += 1
                        if self.socketIO is not None:
                            file_name = 'Line' + str(countLines)
                            file_name = self.saveImage(file_name, line)
                            self.socketIO.start_background_task(self.sendData(file_name, 'Line Sample'))

                        writer_lines.append(line)
                    foundALine = False
        return writer_lines

    def sendData(self, url, label):
        print("SendData")
        self.socketIO.emit('LIWI', {'url': url, 'label': label})

    def makeTempDirectory(self):
        try:
            os.makedirs('D:/Uni/Graduation Project/LIWI/temp')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    def saveImage(self, file_name, image):
        millis = int(round(time.time() * 1000))
        cv2.imwrite('D:/Uni/Graduation Project/LIWI/temp/' + file_name + str(millis) + '.png', image)
        return 'D:/Uni/Graduation Project/LIWI/temp/' + file_name + str(millis) + '.png'
