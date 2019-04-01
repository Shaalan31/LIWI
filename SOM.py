import numpy as np
from neupy import algorithms, utils
import h5py


def on_epoch_end(optimizer):
    print("Last epoch: {}".format(optimizer.last_epoch))

def codebook_generation():

    sofm = algorithms.SOFM(n_inputs = 128,
                                 n_outputs = 300,
                                 step = 0.1,
                                 learning_radius = 0,
                                 signals=on_epoch_end
                                 )
    with h5py.File('SDpoints.h5', 'r') as hf:
        data = hf['keypoints-of-Iam'][:]
    sofm.train(data,epoch=1)


codebook_generation()
