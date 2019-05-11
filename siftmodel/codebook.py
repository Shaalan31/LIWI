import numpy as np
from neupy import algorithms, utils, storage
import h5py
import glob
import cv2 as cv
from siftmodel.sift import *
from utils import *
import pickle


def on_epoch_end(self, optimizer):
    print("Last epoch: {}".format(optimizer.last_epoch))
    storage.load(optimizer, filepath='file.hdf5')


class Som_iam:

    def __init__(self,config_file_path):
        config_file = open(config_file_path, "r")
        #Array of configurations
        config = config_file.read().split(',')

        self.descriptors = None
        self.sofm = None
        self.data = None



    # load descriptors if already done before
    def load_descriptors(self,filename):
        with open(filename, 'rb') as input:
            self.descriptors = pickle.load(input)


    def init_sofm(self,lr=0.5):
        self.sofm = algorithms.SOFM(n_inputs=128,n_outputs=300,step=lr,learning_radius=0,weight='init_pca',shuffle_data=True,verbose =True)
        with open('sofm_iam.pkl', 'wb') as output:
            pickle.dump(self.sofm, output, pickle.HIGHEST_PROTOCOL)

    def read_sofm(self):
        with open('sofm_iam.pkl', 'rb') as input:
            self.sofm = pickle.load(input)

    def train_sofm(self,data,ep=1):
        self.sofm.train(data,epochs=ep)
        with open('sofm_iam.pkl', 'wb') as output:
            pickle.dump(self.sofm, output, pickle.HIGHEST_PROTOCOL)


    def generate_codebook(self,sofm):
        centers=(sofm.weight).transpose()
        with open('centers_iam.pkl', 'wb') as output:
            pickle.dump(centers, output, pickle.HIGHEST_PROTOCOL)
        print(centers)
        print("centers shape are : ",centers.shape)


    def read_codebook(self):
        with open('centers_iam.pkl', 'rb') as input:
            centers = pickle.load(input)
        return centers


    # def codebook_generation(self,num_batches, sofm, epoch):
    #
    #     if (sofm is None):
    #         sofm = algorithms.SOFM(n_inputs=128,
    #                                n_outputs=300,
    #                                step=0.5,
    #                                learning_radius=0,
    #                                signals=on_epoch_end
    #                                )
    #
    #     with h5py.File('Datasets/SDpoints0.h5', 'r') as hf:
    #         data = hf['keypoints-batch'][:]
    #     for x in range(1, int(num_batches) + 1):
    #         with h5py.File('Datasets/SDpoints0.h5', 'r') as hf:
    #             data = np.append(data, hf['keypoints-batch'][:], axis=0)
    #     sofm.train(data, epochs=int(epoch))

    # def data_preprocessor(self):
    #     sift = Sift()
    #     with open("Output.txt", 'w') as out:
    #         out.write("")
    #     batch_num = 0
    #     SDpoints = np.zeros((1, 128))
    #     for filename in glob.glob('WordsDatabase/*/*/*.png'):
    #         temp = sift.get_des(cv.imread(filename))
    #         if temp is not None:
    #             SDpoints = np.append(SDpoints, temp, axis=0)
    #         if SDpoints.shape[0] > 40000:
    #             SDpoints = np.delete(SDpoints, (0), axis=0)
    #             # SDpoints, _,_ = feature_normalize(SDpoints)
    #             with h5py.File('Datasets/SDpoints' + str(batch_num) + '.h5', 'w') as hf:
    #                 hf.create_dataset("keypoints-batch", data=SDpoints)
    #
    #             with open("Output.txt", 'a') as out:
    #                 out.write(str(batch_num) + "   " + filename + "    " + str(SDpoints.shape[0]) + "\n")
    #             # print(str(batch_num) + "   " +filename + "    " + str(SDpoints.shape[0]) + "\n")
    #             batch_num += 1
    #             SDpoints = np.zeros((1, 128))
    #     with h5py.File('Datasets/SDpoints' + str(batch_num) + '.h5', 'w') as hf:
    #         hf.create_dataset("keypoints-batch", data=SDpoints)
    #
    #     with open("Output.txt", 'a') as out:
    #         out.write(str(batch_num) + "   " + filename + "    " + str(SDpoints.shape[0]) + "\n")
    #     print(str(batch_num) + "   " + filename + "    " + str(SDpoints.shape[0]) + "\n")
    #     batch_num += 1

    def normalize(self):
        with open("stats.txt", 'w') as out:
            out.write("")
        SDpoints = np.zeros((1, 128))
        SDpointsFinal = np.zeros((1, 128))
        for x in range(0, 244):
            print(x)
            with h5py.File('Datasets/SDpoints' + str(x) + '.h5', 'r') as hf:
                SDpoints = np.append(SDpoints, hf['keypoints-batch'][:], axis=0)
            if (x % 50 == 0):
                SDpoints = np.delete(SDpoints, (0), axis=0)
                SDpointsFinal = np.append(SDpointsFinal, SDpoints, axis=0)
                print(SDpointsFinal.shape)
                SDpoints = np.zeros((1, 128))
        SDpoints = np.delete(SDpoints, (0), axis=0)
        SDpointsFinal = np.append(SDpointsFinal, SDpoints, axis=0)
        SDpoints = np.zeros((1, 128))
        # print(SDpointsFinal.shape)
        SDpointsFinal = np.delete(SDpointsFinal, (0), axis=0)
        # SDpointsFinal, mean,dev = feature_normalize(SDpointsFinal)

        mean = np.mean(SDpointsFinal, axis=0)
        print(SDpointsFinal)
        mean = mean.reshape((1, 128))
        print(SDpointsFinal.transpose().shape, SDpointsFinal.shape)
        normalized_X = SDpointsFinal - mean
        print(normalized_X.shape)
        deviation = np.sqrt(np.var(normalized_X, axis=0))
        normalized_X = np.divide(normalized_X, deviation)
        with open("stats.txt", 'a') as out:
            out.write("Mean:  " + str(mean) + " \nDev: " + str(deviation))
        with h5py.File('Datasets/AData.h5', 'w') as hf:
            hf.create_dataset("keypoints-batch", data=normalized_X)




class Som_KHATT:

    def __init__(self,new = True):

        self.descriptors = None
        self.load_descriptors()
        print(self.descriptors.shape)
        self.sofm = None
        self.centers = None
        if not new:
            self.init_sofm()
        else:
            self.read_sofm()
        print(self.sofm)


    # load descriptors if already done before
    def load_descriptors(self,filename = 'C:/Users/omars/Documents/Github/LIWI/Datasets/KHATT/AData.h5' ):
        with h5py.File(filename, 'r') as hf:
            self.descriptors = hf['keypoints-batch'][:]


    def init_sofm(self,lr=0.5):
        self.sofm = algorithms.SOFM(n_inputs=128,n_outputs=300,step=lr,learning_radius=0,weight='sample_from_data',shuffle_data=True,verbose =True)
        with open('sofm_KHATT1.pkl', 'wb') as output:
            pickle.dump(self.sofm, output, pickle.HIGHEST_PROTOCOL)

    def read_sofm(self):
        with open('sofm_KHATT.pkl', 'rb') as input:
            self.sofm = pickle.load(input)

    def train_sofm(self,ep=1):
        self.sofm.train(self.descriptors,epochs=ep)
        with open('sofm_KHATT.pkl', 'wb') as output:
            pickle.dump(self.sofm, output, pickle.HIGHEST_PROTOCOL)


    def generate_codebook(self):
        self.centers=(self.sofm.weight).transpose()
        with open('centers_KHATT.pkl', 'wb') as output:
            pickle.dump(self.centers, output, pickle.HIGHEST_PROTOCOL)
        print(self.centers)
        print("centers shape are : ",self.centers.shape)


    def read_codebook(self):
        with open('centers_KHATT.pkl', 'rb') as input:
            self.centers = pickle.load(input)


    def train_loop(self,ep=1):
        self.train_sofm(ep)
        self.generate_codebook()

    # def codebook_generation(self,num_batches, sofm, epoch):
    #
    #     if (sofm is None):
    #         sofm = algorithms.SOFM(n_inputs=128,
    #                                n_outputs=300,
    #                                step=0.5,
    #                                learning_radius=0,
    #                                signals=on_epoch_end
    #                                )
    #
    #     with h5py.File('Datasets/SDpoints0.h5', 'r') as hf:
    #         data = hf['keypoints-batch'][:]
    #     for x in range(1, int(num_batches) + 1):
    #         with h5py.File('Datasets/SDpoints0.h5', 'r') as hf:
    #             data = np.append(data, hf['keypoints-batch'][:], axis=0)
    #     sofm.train(data, epochs=int(epoch))

    # def data_preprocessor(self):
    #     sift = Sift()
    #     with open("Output.txt", 'w') as out:
    #         out.write("")
    #     batch_num = 0
    #     SDpoints = np.zeros((1, 128))
    #     for filename in glob.glob('WordsDatabase/*/*/*.png'):
    #         temp = sift.get_des(cv.imread(filename))
    #         if temp is not None:
    #             SDpoints = np.append(SDpoints, temp, axis=0)
    #         if SDpoints.shape[0] > 40000:
    #             SDpoints = np.delete(SDpoints, (0), axis=0)
    #             # SDpoints, _,_ = feature_normalize(SDpoints)
    #             with h5py.File('Datasets/SDpoints' + str(batch_num) + '.h5', 'w') as hf:
    #                 hf.create_dataset("keypoints-batch", data=SDpoints)
    #
    #             with open("Output.txt", 'a') as out:
    #                 out.write(str(batch_num) + "   " + filename + "    " + str(SDpoints.shape[0]) + "\n")
    #             # print(str(batch_num) + "   " +filename + "    " + str(SDpoints.shape[0]) + "\n")
    #             batch_num += 1
    #             SDpoints = np.zeros((1, 128))
    #     with h5py.File('Datasets/SDpoints' + str(batch_num) + '.h5', 'w') as hf:
    #         hf.create_dataset("keypoints-batch", data=SDpoints)
    #
    #     with open("Output.txt", 'a') as out:
    #         out.write(str(batch_num) + "   " + filename + "    " + str(SDpoints.shape[0]) + "\n")
    #     print(str(batch_num) + "   " + filename + "    " + str(SDpoints.shape[0]) + "\n")
    #     batch_num += 1

    def normalize(self):
        with open("stats.txt", 'w') as out:
            out.write("")
        SDpoints = np.zeros((1, 128))
        SDpointsFinal = np.zeros((1, 128))
        for x in range(0, 244):
            print(x)
            with h5py.File('Datasets/SDpoints' + str(x) + '.h5', 'r') as hf:
                SDpoints = np.append(SDpoints, hf['keypoints-batch'][:], axis=0)
            if (x % 50 == 0):
                SDpoints = np.delete(SDpoints, (0), axis=0)
                SDpointsFinal = np.append(SDpointsFinal, SDpoints, axis=0)
                print(SDpointsFinal.shape)
                SDpoints = np.zeros((1, 128))
        SDpoints = np.delete(SDpoints, (0), axis=0)
        SDpointsFinal = np.append(SDpointsFinal, SDpoints, axis=0)
        SDpoints = np.zeros((1, 128))
        # print(SDpointsFinal.shape)
        SDpointsFinal = np.delete(SDpointsFinal, (0), axis=0)
        # SDpointsFinal, mean,dev = feature_normalize(SDpointsFinal)

        mean = np.mean(SDpointsFinal, axis=0)
        print(SDpointsFinal)
        mean = mean.reshape((1, 128))
        print(SDpointsFinal.transpose().shape, SDpointsFinal.shape)
        normalized_X = SDpointsFinal - mean
        print(normalized_X.shape)
        deviation = np.sqrt(np.var(normalized_X, axis=0))
        normalized_X = np.divide(normalized_X, deviation)
        with open("stats.txt", 'a') as out:
            out.write("Mean:  " + str(mean) + " \nDev: " + str(deviation))
        with h5py.File('Datasets/AData.h5', 'w') as hf:
            hf.create_dataset("keypoints-batch", data=normalized_X)
