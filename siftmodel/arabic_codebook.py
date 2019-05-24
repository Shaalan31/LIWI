from siftmodel.codebook import *


SOFM = Som_KHATT()



# SOFM.generate_codebook()
for x in range(100):
    print(x)
    SOFM.train_loop()




#
# with open('centers_KHATT.pkl', 'rb') as input:
#     centers = pickle.load(input)
#
# mean = np.genfromtxt('C:/Users/omars/Documents/Github/LIWI/Datasets/KHATT/mean.csv', delimiter=",").reshape((1,128))
# deviation = np.genfromtxt('C:/Users/omars/Documents/Github/LIWI/Datasets/KHATT/deviation.csv', delimiter=",").reshape(1,128)
#
# new_centers = (centers * deviation) + mean
# print(new_centers.shape)
#
# with open('new_centers_KHATT.pkl', 'wb') as output:
#     pickle.dump(new_centers, output, pickle.HIGHEST_PROTOCOL)