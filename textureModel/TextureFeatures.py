#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) Philipp Wagner. All rights reserved.
# Licensed under the BSD license. See LICENSE file in the project root for full license information.

import numpy as np
from skimage.feature import local_binary_pattern
from scipy.signal import convolve2d


class LocalDescriptor(object):
    def __init__(self, neighbors):
        self._neighbors = neighbors

    def __call__(self, X):
        raise NotImplementedError("Every LBPOperator must implement the __call__ method.")

    @property
    def neighbors(self):
        return self._neighbors

    def __repr__(self):
        return "LBPOperator (neighbors=%s)" % (self._neighbors)


class LPQ(LocalDescriptor):
    """ This implementation of Local Phase Quantization (LPQ) is a 1:1 adaption of the
        original implementation by Ojansivu V & Heikkilä J, which is available at:

            * http://www.cse.oulu.fi/CMV/Downloads/LPQMatlab

        So all credit goes to them.

      Reference:
        Ojansivu V & Heikkilä J (2008) Blur insensitive texture classification
        using local phase quantization. Proc. Image and Signal Processing
        (ICISP 2008), Cherbourg-Octeville, France, 5099:236-243.
        Copyright 2008 by Heikkilä & Ojansivu
    """

    def __init__(self, radius=3):
        LocalDescriptor.__init__(self, neighbors=8)
        self._radius = radius

    def euc_dist(self, X):
        Y = X = X.astype(np.float)
        XX = np.sum(X * X, axis=1)[:, np.newaxis]
        YY = XX.T
        distances = np.dot(X, Y.T)
        distances *= -2
        distances += XX
        distances += YY
        np.maximum(distances, 0, distances)
        distances.flat[::distances.shape[0] + 1] = 0.0
        return np.sqrt(distances)

    def __call__(self, X):
        f = 1.0
        x = np.arange(-self._radius, self._radius + 1)
        n = len(x)
        rho = 0.95
        [xp, yp] = np.meshgrid(np.arange(1, (n + 1)), np.arange(1, (n + 1)))
        pp = np.concatenate((xp, yp)).reshape(2, -1)
        dd = self.euc_dist(pp.T)  # squareform(pdist(...)) would do the job, too...
        C = np.power(rho, dd)

        w0 = (x * 0.0 + 1.0)
        w1 = np.exp(-2 * np.pi * 1j * x * f / n)
        w2 = np.conj(w1)

        q1 = w0.reshape(-1, 1) * w1
        q2 = w1.reshape(-1, 1) * w0
        q3 = w1.reshape(-1, 1) * w1
        q4 = w1.reshape(-1, 1) * w2

        u1 = np.real(q1)
        u2 = np.imag(q1)
        u3 = np.real(q2)
        u4 = np.imag(q2)
        u5 = np.real(q3)
        u6 = np.imag(q3)
        u7 = np.real(q4)
        u8 = np.imag(q4)

        M = np.matrix(
            [u1.flatten(1), u2.flatten(1), u3.flatten(1), u4.flatten(1), u5.flatten(1), u6.flatten(1), u7.flatten(1),
             u8.flatten(1)])

        D = np.dot(np.dot(M, C), M.T)
        U, S, V = np.linalg.svd(D)

        Qa = convolve2d(convolve2d(X, w0.reshape(-1, 1), mode='same'), w1.reshape(1, -1), mode='same')
        Qb = convolve2d(convolve2d(X, w1.reshape(-1, 1), mode='same'), w0.reshape(1, -1), mode='same')
        Qc = convolve2d(convolve2d(X, w1.reshape(-1, 1), mode='same'), w1.reshape(1, -1), mode='same')
        Qd = convolve2d(convolve2d(X, w1.reshape(-1, 1), mode='same'), w2.reshape(1, -1), mode='same')

        Fa = np.real(Qa)
        Ga = np.imag(Qa)
        Fb = np.real(Qb)
        Gb = np.imag(Qb)
        Fc = np.real(Qc)
        Gc = np.imag(Qc)
        Fd = np.real(Qd)
        Gd = np.imag(Qd)

        F = np.array(
            [Fa.flatten(1), Ga.flatten(1), Fb.flatten(1), Gb.flatten(1), Fc.flatten(1), Gc.flatten(1), Fd.flatten(1),
             Gd.flatten(1)])
        G = np.dot(V.T, F)

        t = 0

        # Calculate the LPQ Patterns:
        B = (G[0, :] >= t) * 1 + (G[1, :] >= t) * 2 + (G[2, :] >= t) * 4 + (G[3, :] >= t) * 8 + (G[4, :] >= t) * 16 + (
                G[5, :] >= t) * 32 + (G[6, :] >= t) * 64 + (G[7, :] >= t) * 128

        return np.reshape(B, np.shape(Fa))

    def getlbp_Features(self, greyScaleImage):
        return local_binary_pattern(greyScaleImage, P=self._neighbors, R=self._radius, method='uniform')

    @property
    def radius(self):
        return self._radius

    def __repr__(self):
        return "LPQ (neighbors=%s, radius=%s)" % (self._neighbors, self._radius)
