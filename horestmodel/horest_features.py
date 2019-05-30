import errno
import math
import os
import threading
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
import skimage.filters as filters
from scipy import signal
from scipy import stats


class HorestFeatures:
    def __init__(self, image, contours, hierarchy, socket=None):
        self.image = image
        self.contours = contours
        self.hierarchy = hierarchy
        self.socketIO = socket

        # needed in blobs detection threading
        self.areas = []
        self.roundness = []
        self.form_factors = []
        self.lock1 = threading.Lock()
        self.lock3 = threading.Lock()
        self.lock4 = threading.Lock()
        self.num_threads = 3
        self.step = self.num_threads

    def angles_histogram(self):
        values, count = np.unique(self.image, return_counts=True)
        countBlack = count[0]

        # filters = Filters()
        # sob_img_v = np.multiply(filters.sobelv(self.image), 255)
        # sob_img_h = np.multiply(filters.sobelh(self.image), 255)
        sob_img_v = np.multiply(filters.sobel_v(self.image), 255)
        sob_img_h = np.multiply(filters.sobel_h(self.image), 255)

        sobelXY = np.sqrt(np.power(sob_img_h, 2) + np.power(sob_img_v, 2))
        self.sendEdgeDetectionSample(sobelXY)

        # Getting angles in radians
        angles = np.arctan2(sob_img_v, sob_img_h)
        angles = np.multiply(angles, (180 / math.pi))
        angles = np.round(angles)

        abs_angles = np.abs(angles)

        anglesHist = []
        abs_angles_hist = []
        angle1 = 10
        angle2 = 40

        while angle2 < 180:
            anglesCopy = angles.copy()
            anglesCopy[np.logical_or(anglesCopy < angle1, anglesCopy > angle2)] = 0
            anglesCopy[np.logical_and(anglesCopy >= angle1, anglesCopy <= angle2)] = 1
            anglesHist.append(np.sum(anglesCopy))

            abs_angles_copy = abs_angles.copy()
            abs_angles_copy[np.logical_or(abs_angles_copy < angle1, abs_angles_copy > angle2)] = 0
            abs_angles_copy[np.logical_and(abs_angles_copy >= angle1, abs_angles_copy <= angle2)] = 1
            abs_angles_hist.append(np.sum(abs_angles_copy))

            angle1 += 30
            angle2 += 30

        hist = np.divide(anglesHist, countBlack)
        abs_hist = np.divide(abs_angles_hist, countBlack)
        self.sendHistogram(abs_hist)
        return hist

    def connected_components(self, image_shape):
        mask = (self.hierarchy[:, 3] == 0).astype('int')
        contours = self.contours[np.where(mask)]
        bounding_rect = np.zeros((len(contours), 6))

        # average h/w
        for i in range(0, len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            bounding_rect[i] = (int(x), int(y), int(w), int(h), int(w * h), int(h / w))
        # h_to_w_ratio
        h_to_w_ratio = np.average(bounding_rect[:, 5], axis=0)

        bounding_rect_sorted = bounding_rect[bounding_rect[:, 0].argsort()]
        iAmDbImageSize = 375 / 8780618
        mask = (bounding_rect_sorted[:, 4] > (iAmDbImageSize * (image_shape[0] * image_shape[1]))).astype('int')
        bounding_rect_sorted = bounding_rect_sorted[np.where(mask)]

        diff_dist_word = np.diff(bounding_rect_sorted, axis=0)[:, 0] - bounding_rect_sorted[:-1, 2]
        threshold = np.average(np.abs(np.diff(bounding_rect_sorted, axis=0)[:, 0] - bounding_rect_sorted[:-1, 2]))
        word_dist = np.average(diff_dist_word[np.where(diff_dist_word > threshold)])
        # if line consists of only one word
        if math.isnan(word_dist):
            word_dist = 0
        within_word_dist = np.average(np.abs(diff_dist_word[np.where(diff_dist_word < threshold)]))
        if math.isnan(within_word_dist):
            within_word_dist = 0
        total_transitions = 0
        img = self.image / 255
        for x, y, w, h, a, r in bounding_rect_sorted:
            total_transitions += np.sum(np.abs(np.diff(img[int(y):int(y + h), int(x):int(x + w)])))
        total_transitions /= ((2 * bounding_rect_sorted.shape[0])+1e-7)
        sdW = np.sqrt(np.var(bounding_rect_sorted[:, 2]))
        MedianW = np.median(bounding_rect_sorted[:, 2])
        AverageW = np.average(bounding_rect_sorted[:, 2])

        return np.asarray([word_dist, within_word_dist, total_transitions, sdW, MedianW, AverageW, h_to_w_ratio])

    def disk_fractal(self, loops=25):
        arr = np.zeros((loops, 2))
        arr[1] = ([np.log(1), np.log(np.sum(255 - self.image) / 255) - np.log(1)])
        for x in range(2, loops):
            img_dilate = cv2.erode(self.image.copy(),
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * x - 1, 2 * x - 1)),
                                   iterations=1)
            arr[x] = ([np.log(x), np.log(np.sum(255 - img_dilate) / 255) - np.log(x)])

        error = 999
        slope = [0, 0, 0]
        loops = int(loops)
        for x in range(2, loops - 2):
            for y in range(x + 2, loops - 1):
                first = arr[1:x + 1, :]
                second = arr[x + 1:y + 1, :]
                third = arr[y + 1:loops, :]
                slope1, _, _, _, std_err1 = stats.linregress(x=first[:, 0], y=first[:, 1])
                slope2, _, _, _, std_err2 = stats.linregress(x=second[:, 0], y=second[:, 1])
                slope3, _, _, _, std_err3 = stats.linregress(x=third[:, 0], y=third[:, 1])

                if error > std_err1 + std_err2 + std_err3:
                    error = std_err1 + std_err2 + std_err3
                    slope = [slope1, slope2, slope3]

        return slope

    def blob_threaded(self):
        global areas
        global roundness
        global form_factors

        mask = (self.hierarchy[:, 3] > 0).astype('int')
        contours = self.contours[np.where(mask)]

        t1 = threading.Thread(target=self.loop_on_contours, args=(contours, 0))
        t2 = threading.Thread(target=self.loop_on_contours, args=(contours, 1))
        t3 = threading.Thread(target=self.loop_on_contours, args=(contours, 2))

        t1.start()
        t2.start()
        t3.start()

        t1.join()
        t2.join()
        t3.join()

        avg_areas = np.average(self.areas)
        avg_roundness = np.average(self.roundness)
        avg_form_factors = np.average(self.form_factors)

        areas = []
        roundness = []
        form_factors = []

        return [avg_areas, avg_roundness, avg_form_factors]

    def loop_on_contours(self, contours, starting_index):
        for i in range(starting_index, len(contours), self.step):
            contour = contours[i]
            current_area = cv2.contourArea(contour)
            if current_area == 0:
                continue
            current_length = cv2.arcLength(contour, True)
            current_length_sq = current_length * current_length

            self.lock1.acquire()
            self.areas.append(current_area)
            self.lock1.release()

            self.lock3.acquire()
            self.form_factors.append(4 * current_area * math.pi / current_length_sq)
            self.lock3.release()

            self.lock4.acquire()
            self.roundness.append(current_length_sq / current_area)
            self.lock4.release()

    def elipse_fractal(self, angle, loops=25):
        arr = np.zeros((loops, 2))
        arr[1] = ([np.log(1), np.log(np.sum(255 - self.img) / 255) - np.log(1)])

        for x in range(2, loops):
            ellipse_mask = ndimage.rotate(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * x - 1, 2 * (2 * x - 1))),
                                          float(angle))
            img_dilate = cv2.erode(self.img.copy(), ellipse_mask,
                                   iterations=1)
            arr[x] = ([np.log(x), np.log(np.sum(255 - img_dilate) / 255) - np.log(x)])

        error = 999
        slope = [0, 0, 0]
        loops = int(loops)
        for x in range(2, loops - 2):
            for y in range(x + 2, loops - 1):
                first = arr[1:x + 1, :]
                second = arr[x + 1:y + 1, :]
                third = arr[y + 1:loops, :]
                slope1, _, _, _, std_err1 = stats.linregress(x=first[:, 0], y=first[:, 1])
                slope2, _, _, _, std_err2 = stats.linregress(x=second[:, 0], y=second[:, 1])
                slope3, _, _, _, std_err3 = stats.linregress(x=third[:, 0], y=third[:, 1])

                if error > std_err1 + std_err2 + std_err3:
                    error = std_err1 + std_err2 + std_err3
                    slope = [slope1, slope2, slope3]

        return slope

    def lower_contour_features(self):
        lower_contour = np.argmin(self.image[::-1], axis=0)
        fig = plt.figure()
        ax = fig.add_subplot('111')
        print(np.max(lower_contour), np.argmax(lower_contour))
        plt.ylim(np.min(lower_contour) - 10, np.max(lower_contour) + 10)
        plt.xlim(0, lower_contour.shape[0] + 10)
        ax.scatter(np.linspace(0, lower_contour.shape[0], lower_contour.shape[0]), lower_contour, s=1)

        mask = self.image[self.image.shape[0] - 1 - lower_contour, np.indices(lower_contour.shape)]
        print(mask)
        print(lower_contour)

        lower_contour = lower_contour[mask[0] == 0]
        diff = np.append(np.asarray([0]), np.diff(lower_contour))

        diff[np.abs(diff) < 2] = 0
        diff[diff < 0] -= 1

        change = np.cumsum(diff)
        print(lower_contour, change, diff)
        lower_contour = lower_contour - change
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            x=np.linspace(0, lower_contour.shape[0], lower_contour.shape[0]), y=lower_contour)
        print(slope, std_err)
        local_max = signal.argrelextrema(lower_contour, np.greater)[0]
        local_min = signal.argrelextrema(lower_contour, np.less)[0]
        max_slope = 0
        min_slope = 0
        minIndex = 0
        for i in local_max:
            if (i < 5):
                minIndex = 0
            else:
                minIndex = i - 5
            slopeMax, _, _, _, _ = stats.linregress(x=np.linspace(0, i - minIndex + 1, i - minIndex + 1),
                                                    y=lower_contour[minIndex:i + 1])
            max_slope += slopeMax
        average_slope_max = max_slope / len(local_max)
        for i in local_min:
            if (i > len(lower_contour) - 5):
                maxIndex = len(lower_contour)
            else:
                maxIndex = i + 5
            slopeMin, _, _, _, _ = stats.linregress(x=np.linspace(0, -i + maxIndex, -i + maxIndex),
                                                    y=lower_contour[i:maxIndex])
            min_slope += slopeMin

        average_slope_min = min_slope / len(local_min)
        max_ratio = len(local_max) / lower_contour.shape[0]
        min_ratio = len(local_min) / lower_contour.shape[0]

        return [slope, std_err, len(local_max), len(local_min), average_slope_max, average_slope_min, max_ratio,
                min_ratio]

    def upper_contour_features(self):
        upper_contour = np.argmin(self.image, axis=0)
        print(upper_contour)
        fig = plt.figure()
        ax = fig.add_subplot('111')
        plt.ylim(np.min(upper_contour) - 10, np.max(upper_contour) + 10)
        plt.xlim(0, upper_contour.shape[0] + 10)
        ax.scatter(np.linspace(0, upper_contour.shape[0], upper_contour.shape[0]), upper_contour, s=1)
        plt.show()
        mask = self.image[upper_contour, np.indices(upper_contour.shape)]
        upper_contour = upper_contour[mask[0] == 0]
        print(upper_contour)

        diff = np.append(np.asarray([0]), np.diff(upper_contour))
        diff[np.abs(diff) < 2] = 0
        change = np.cumsum(diff)
        upper_contour = upper_contour - change
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            x=np.linspace(0, upper_contour.shape[0], upper_contour.shape[0]), y=upper_contour)
        print(slope, std_err)
        local_max = signal.argrelextrema(upper_contour, np.greater)[0]
        local_min = signal.argrelextrema(upper_contour, np.less)[0]
        max_slope = 0
        min_slope = 0
        minIndex = 0
        for i in local_max:
            if (i < 5):
                minIndex = 0
            else:
                minIndex = i - 5
            slopeMax, _, _, _, _ = stats.linregress(x=np.linspace(0, i - minIndex + 1, i - minIndex + 1),
                                                    y=upper_contour[minIndex:i + 1])
            max_slope += slopeMax
        average_slope_max = max_slope / len(local_max)
        for i in local_min:
            if (i > len(upper_contour) - 5):
                maxIndex = len(upper_contour)
            else:
                maxIndex = i + 5
            slopeMin, _, _, _, _ = stats.linregress(x=np.linspace(0, -i + maxIndex, -i + maxIndex),
                                                    y=upper_contour[i:maxIndex])
            min_slope += slopeMin

        average_slope_min = min_slope / len(local_min)
        max_ratio = len(local_max) / upper_contour.shape[0]
        min_ratio = len(local_min) / upper_contour.shape[0]

        return [slope, std_err, len(local_max), len(local_min), average_slope_max, average_slope_min, max_ratio,
                min_ratio]

    def makeTempDirectory(self):
        try:
            os.makedirs('D:/Uni/Graduation Project/LIWI/temp')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    def saveEdgeDetection(self, file_name, image):
        millis = int(round(time.time() * 1000))
        cv2.imwrite('D:/Uni/Graduation Project/LIWI/temp/' + file_name + str(millis) + '.png', image)
        return 'D:/Uni/Graduation Project/LIWI/temp/' + file_name + str(millis) + '.png'

    def saveHistogram(self, _plt, file_name):
        millis = int(round(time.time() * 1000))
        _plt.savefig('D:/Uni/Graduation Project/LIWI/temp/' + file_name + str(millis) + '.png')
        return 'D:/Uni/Graduation Project/LIWI/temp/' + file_name + str(millis) + '.png'

    def sendData(self, url, label):
        print("SendData Angles Histogram")
        self.socketIO.emit('LIWI', {'url': url, 'label': label})

    def sendEdgeDetectionSample(self, img):
        # Khairy (sending edge detection)
        if self.socketIO is not None:
            self.makeTempDirectory()
            file_name = 'EdgeDetection'
            file_name = self.saveEdgeDetection(file_name, img)
            self.socketIO.start_background_task(self.sendData(file_name, 'Edge Detection'))

    def sendHistogram(self, hist):
        if self.socketIO is not None:
            pltRange = np.arange(start=10, stop=160, step=30)
            plt.figure()
            plt.title('Weighted Angles Histogram')
            plt.bar(pltRange.copy(), np.around(np.multiply(hist, 100)).astype('uint8').copy(), width=2, align='center')

            self.makeTempDirectory()
            file_name = self.saveHistogram(plt, 'Angles Histogram')
            self.socketIO.start_background_task(self.sendData(file_name, 'Angles Histogram'))
