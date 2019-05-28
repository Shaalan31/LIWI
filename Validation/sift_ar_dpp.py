#dpp means data pre processor for sift

from siftmodel.sift_model import *
import os, errno
from server.utils.utilities import *


def Samples_gen(start,end):
    print('SAMPPLESS - ',start)
    t = [1, 50, 150 ]
    phi = [36, 72, 108]

    sift_model = SiftModel()
    sift_model.set_code_book('ar')

    start_class = int(start)
    num_classes = int(end)

    base_path = 'C:/Users/omars/Documents/Github/LIWI/Omar/KHATT/Samples/Class'

    base_samples_t = 'C:/Users/omars/Documents/Github/LIWI/Omar/ValidationArabic/Samples/SDS/'
    base_samples_phi = 'C:/Users/omars/Documents/Github/LIWI/Omar/ValidationArabic/Samples/SOH/'




    for class_number in range(start_class, num_classes + 1):


        writer_texture_features = []
        SDS_train = []
        SOH_train = []
        print('Class' + str(class_number) + ':')

        # loop on training data for each writer
        for filename in glob.glob(
                base_path + str(class_number) + '/*.tif'):
            print(filename)

            image = cv2.imread(filename)

            name = Path(filename).name
            name = name.replace('tif','csv')

            print('Sift Model')

            for idx in range(0,3):
                print(idx)
                SDS, SOH = sift_model.get_features(name, image=image,t=t[idx],phi=phi[idx])

                str_t = str(t[idx])
                str_phi = str(phi[idx])

                while len(str_t) < 3:
                    str_t = '0' + str_t

                while len(str_phi) < 3:
                    str_phi = '0' + str_phi

                try:
                    # print(base_samples_h+str(h_coeff)+"/Class"+str(class_number))
                    os.makedirs(base_samples_t + str_t + "/Class" + str(class_number))
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise

                np.savetxt(base_samples_t+str_t+"/Class"+str(class_number)+'/'+name, SDS, delimiter=",")

                try:
                    # print(base_samples_h+str(h_coeff)+"/Class"+str(class_number))
                    os.makedirs(base_samples_phi + str_phi + "/Class" + str(class_number))
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
                np.savetxt(base_samples_phi+str_phi+"/Class"+str(class_number)+'/'+name, SOH, delimiter=",")







def Testcase_gen(start,num):
    t = [1, 50, 150]
    phi = [36, 72, 108]

    sift_model = SiftModel()
    sift_model.set_code_book('ar')


    base_path = 'C:/Users/omars/Documents/Github/LIWI/Omar/KHATT/TestCases/'

    base_samples_t = 'C:/Users/omars/Documents/Github/LIWI/Omar/ValidationArabic/Samples/SDS/'
    base_samples_phi = 'C:/Users/omars/Documents/Github/LIWI/Omar/ValidationArabic/Samples/SOH/'


    print('TESTCASES - ',start)



    start_class = int(start)
    num_classes = int(num)

    base_test_t = 'C:/Users/omars/Documents/Github/LIWI/Omar/ValidationArabic/TestCases/SDS/'
    base_test_phi = 'C:/Users/omars/Documents/Github/LIWI/Omar/ValidationArabic/TestCases/SOH/'


    for class_number in range(start_class, num_classes + 1):

        SDS_train = []
        SOH_train = []
        print('Class' + str(class_number) + ':')

        # loop on training data for each writer

        for filename in glob.glob(
                base_path + 'testing'+str(class_number) + '.png'):
            print(filename)

            image = cv2.imread(filename)

            name = Path(filename).name
            name = name.replace('png','csv')
            print(name)
            print('Sift Model')

            for idx in range(0,3):
                print(idx)
                SDS, SOH = sift_model.get_features(name, image=image,t=t[idx],phi=phi[idx])

                str_t = str(t[idx])
                str_phi = str(phi[idx])

                while len(str_t) < 3:
                    str_t = '0' + str_t

                while len(str_phi) < 3:
                    str_phi = '0' + str_phi

                try:

                    os.makedirs(base_test_t+str_t)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise

                try:

                    os.makedirs(base_test_phi + str_phi)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise

                np.savetxt(base_test_t+str_t+'/'+name, SDS, delimiter=",")
                np.savetxt(base_test_phi+str_phi+'/'+name, SOH, delimiter=",")





for beg in range(170,350,20):
    Testcase_gen(beg,20+beg)
    Samples_gen(beg,20+beg)