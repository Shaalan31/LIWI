import numpy as np
import math


class FeaturesExtraction:
    def __init__(self):
        pass

    def sds(self, des, code_book_center, t):
        SDS = np.zeros((1, 300))
        for key,word in des.items():
            if(word is None):
                continue
            for point in word:
                ED = np.subtract(code_book_center , point)
                ED = np.square(ED)
                ED = np.sum(ED, axis=1).reshape((1, 300))
                EDV = np.sqrt(ED).reshape((1, 300))
                IDX = np.argsort(EDV).reshape(1, 300)[0, :int(t)]
                SDS[0, IDX] += 1
        return np.divide(SDS, np.sum(SDS)).reshape((1, 300))

    def soh(self, key_points, phi):
        Obin = math.ceil(360 / phi)

        key_points[key_points[:, 2] == 0, 2] = key_points[key_points[:, 2] == 0, 2] + 1e-16

        scales = ((key_points[:, 1] - np.min(key_points[:, 1])) * np.max(key_points[:, 2]) + key_points[:, 2])
        scales = scales.reshape((1,scales.shape[0]))

        # compute index idx in SOH feature vector
        bin = np.ceil(key_points[:, 0]/phi)
        bin = bin.reshape((1,bin.shape[0]))
        idx = np.int32(Obin*(scales - 1) + bin) - 1
        real_idx = np.unique(idx,return_counts=True)
        idxx = real_idx[0].reshape((1,real_idx[0].shape[0]))
        count = real_idx[1].reshape((1,real_idx[0].shape[0]))

        # update the SOH feature vector
        SOH = np.zeros((1, int(Obin * np.max(scales))))
        SOH[0,idxx] =  count[0, np.indices((1, real_idx[0].shape[0]))[:, 0][1]]

        SOH = np.divide(SOH, np.sum(SOH))

        return SOH.reshape((1, SOH.shape[1]))
