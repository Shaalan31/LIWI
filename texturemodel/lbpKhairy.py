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
                rangeToTraverse = range(0, length)
                if fx == length - 1:
                    rangeToTraverse = range(length - 1, -1, -1)
                for fy in rangeToTraverse:
                    if (fx - edgex) == 0 and (fy - edgey) == 0:
                        continue
                    comparison = 1
                    if image[i][j] > image[i + fx - edgex][j + fy - edgey]:
                        comparison = 0
                    pattern += str(comparison)

            pattern, lastChar = removeCharByIndexAndReturnChar(pattern, 5)
            pattern += lastChar
            lbpValues = np.append(lbpValues, int(str(pattern), 2))

    return lbpValues


def removeCharByIndexAndReturnChar(string, index):
    lastChar = string[index]
    string = string[:index] + string[index+1:]
    return string, lastChar

#
# image = np.asarray([[195, 153, 200], [200, 128, 33], [18, 81, 201]])
# val = getLBP(image, 3)
# print(val)
