import numpy as np
from neupy import algorithms, utils
import h5py


def on_epoch_end(optimizer):
    print("Last epoch: {}".format(optimizer.last_epoch))

def codebook_generation():

    sofm = algorithms.SOFM(n_inputs = 128,
                                 n_outputs = 300 ,
                                 step = 0.5,
                                 learning_radius = 0,
                                 signals=on_epoch_end
                                 )

    with h5py.File('Datasets/SDpoints1.h5', 'r') as hf:
        data = hf['keypoints-batch'][:]

    print(data.shape)
    #print(sofm.weight)
    #sofm.init_weights(data)

    #for x in range(1000):
    sofm.train(data,epochs=5)
    #print(sofm.predict(np.array([2])))
    print(sofm.weight.shape)
    #sofm.train(data,epoch=1)


codebook_generation()
