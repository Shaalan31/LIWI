import numpy as np
from neupy import algorithms, utils, storage
import h5py


import pickle


def on_epoch_end(optimizer):
    print("Last epoch: {}".format(optimizer.last_epoch))
    storage.load(optimizer, filepath='file.hdf5')


def train_sofm(data,ep=1):
    with open('sofm.pkl','rb') as input:
        sofm = pickle.load(input)
    sofm.train(data,epochs=ep)
    with open('sofm.pkl','wb') as output:
        pickle.dump(sofm,output,pickle.HIGHEST_PROTOCOL)

    return sofm


def codebook_generation(num_batches,sofm,epoch):

    if(sofm is None):
        sofm = algorithms.SOFM(n_inputs = 128,
                                     n_outputs = 300 ,
                                     step = 0.5,
                                     learning_radius = 0,
                                     signals=on_epoch_end
                                     )


    with h5py.File('Datasets/SDpoints0.h5', 'r') as hf:
        data = hf['keypoints-batch'][:]
    for x in range(1,int(num_batches)+1):
        with h5py.File('Datasets/SDpoints0.h5', 'r') as hf:
            data = np.append(data,hf['keypoints-batch'][:],axis=0)
    sofm.train(data,epochs=int(epoch))


codebook_generation(12,None,5)


