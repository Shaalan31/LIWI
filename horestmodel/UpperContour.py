import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import signal

def LowerContourFeatures(img):
    lower_contour = np.argmin(img[::-1], axis=0)
    fig = plt.figure()
    ax = fig.add_subplot('111')
    print(np.max(lower_contour), np.argmax(lower_contour))
    plt.ylim(np.min(lower_contour) - 10, np.max(lower_contour) + 10)
    plt.xlim(0, lower_contour.shape[0] + 10)
    ax.scatter(np.linspace(0, lower_contour.shape[0], lower_contour.shape[0]), lower_contour, s=1)

    mask = img[img.shape[0] - 1 - lower_contour, np.indices(lower_contour.shape)]
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
    average_slope_max=max_slope/len(local_max)
    for i in local_min:
        if (i > len(lower_contour) - 5):
            maxIndex = len(lower_contour)
        else:
            maxIndex = i + 5
        slopeMin, _, _, _, _ = stats.linregress(x=np.linspace(0, -i + maxIndex, -i + maxIndex),
                                                y=lower_contour[i:maxIndex])
        min_slope += slopeMin

    average_slope_min=min_slope/len(local_min)
    max_ratio = len(local_max) / lower_contour.shape[0]
    min_ratio = len(local_min) / lower_contour.shape[0]

    return [slope,std_err,len(local_max),len(local_min),average_slope_max,average_slope_min,max_ratio,min_ratio]


def UpperContourFeatures(img):
    upper_contour = np.argmin(img, axis=0)
    print(upper_contour)
    fig = plt.figure()
    ax = fig.add_subplot('111')
    plt.ylim(np.min(upper_contour) - 10, np.max(upper_contour) + 10)
    plt.xlim(0, upper_contour.shape[0] + 10)
    ax.scatter(np.linspace(0, upper_contour.shape[0], upper_contour.shape[0]), upper_contour, s=1)
    plt.show()
    mask = img[upper_contour, np.indices(upper_contour.shape)]
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
    average_slope_max=max_slope/len(local_max)
    for i in local_min:
        if (i > len(upper_contour) - 5):
            maxIndex = len(upper_contour)
        else:
            maxIndex = i + 5
        slopeMin, _, _, _, _ = stats.linregress(x=np.linspace(0, -i + maxIndex, -i + maxIndex),
                                                y=upper_contour[i:maxIndex])
        min_slope += slopeMin

    average_slope_min=min_slope/len(local_min)
    max_ratio = len(local_max) / upper_contour.shape[0]
    min_ratio = len(local_min) / upper_contour.shape[0]

    return [slope,std_err,len(local_max),len(local_min),average_slope_max,average_slope_min,max_ratio,min_ratio]

