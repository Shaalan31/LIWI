import numpy as np
import math

def sds(des, code_book_center, t):
    SDS = np.zeros((1, 300))
    for key,word in des.items():
        for point in word:
            ED = np.subtract(code_book_center , point)
            ED = np.square(ED)
            ED = np.sum(ED, axis=1).reshape((1, 300))
            EDV = np.sqrt(ED).reshape((1, 300))
            IDX = np.argsort(EDV).reshape(1, 300)[0, :int(t)]
            SDS[0, IDX] += 1
    return np.divide(SDS, np.sum(SDS)).reshape((1, 300))


def soh(key_points, phi):
    Obin = math.ceil(360 / phi)


    return