import numpy as np
from neupy import algorithms, utils
import h5py


def on_epoch_end(optimizer):
    print("Last epoch: {}".format(optimizer.last_epoch))

def codebook_generation():

    sofm = algorithms.SOFM(n_inputs = 1,
                                 n_outputs = 4 ,
                                 step = 0.5,
                                 learning_radius = 0
                                 #signals=on_epoch_end
                                 )
    data = np.zeros((4,1))
    data[0] = 0
    data[1] = 1
    data[2] = 2
    data[3] = 3
    print(data)
    print(sofm.weight)
    #sofm.init_weights(data)

    #with h5py.File('SDpoints.h5', 'r') as hf:
    #    data = hf['keypoints-of-Iam'][:]
    #for x in range(1000):
    sofm.train(data,epochs=10000)
    #print(sofm.predict(np.array([2])))
    print(sofm.weight)
    #sofm.train(data,epoch=1)


codebook_generation()
