import math
import numpy as np


def getLBP(image, length):
    edgex = edgey = int(math.floor(length / 2))
    imageWidth = image.shape[1]
    imageHeight = image.shape[0]

    lbpValues = np.asarray([])

    for j in range(edgex, imageWidth - edgex):
        for i in range(edgey, imageHeight - edgey):
            pattern = str('0b')
            for fx in range(0, length):
                for fy in range(0, length):
                    if (fx - edgex) == 0 and (fy - edgey) == 0:
                        continue
                    comparison = 1
                    if image[i][j] > image[i + fx - edgex][j + fy - edgey]:
                        comparison = 0
                    pattern += str(comparison)
            lbpValues = np.append(lbpValues, int(str(pattern), 2))

    return lbpValues
