from texturemodel.texture_model import *
from siftmodel.sift_model import *
import os, errno
from server.utils.utilities import *




def Samples_gen(start,num):
    print('SAMPPLESS - ',start)
    t = [1, 50, 150, 225, 300]
    phi = [36, 72, 108, 234, 360]
    h = [0.1, 0.3, 0.5, 0.7, 0.9]

    texture_model = TextureWriterIdentification()
    sift_model = SiftModel()
    sift_model.set_code_book('en')

    start_class = int(start)
    num_classes = int(num)

    base_path = 'C:/Users/omars/Documents/Github/LIWI/Omar/Dataset/Training/Class'
    base_samples_h = 'C:/Users/omars/Documents/Github/LIWI/Omar/Validation/Samples/H/'
    base_samples_t = 'C:/Users/omars/Documents/Github/LIWI/Omar/Validation/Samples/SDS/'
    base_samples_phi = 'C:/Users/omars/Documents/Github/LIWI/Omar/Validation/Samples/SOH/'

    base_test_h = 'C:/Users/omars/Documents/Github/LIWI/Omar/Validation/TestCases/H/'
    base_test_t = 'C:/Users/omars/Documents/Github/LIWI/Omar/Validation/TestCases/SDS/'
    base_test_phi = 'C:/Users/omars/Documents/Github/LIWI/Omar/Validation/TestCases/SOH/'


    for class_number in range(start_class, num_classes + 1):


        writer_texture_features = []
        SDS_train = []
        SOH_train = []
        texture_model.num_blocks_per_class = 0
        print('Class' + str(class_number) + ':')

        # loop on training data for each writer
        for filename in glob.glob(
                base_path + str(class_number) + '/*.jpg'):
            print(filename)

            image = cv2.imread(filename)

            name = Path(filename).name
            name = name.replace('jpg','csv')
            print('Texture Features')
            # writer_texture_features.append(texture_model.get_features(cv2.imread(filename))[0].tolist())

            for h_coeff in h:
                print(h_coeff)
                _, texture_features = texture_model.get_features(image)
                writer_texture_features = np.append(writer_texture_features, texture_features[0].tolist())
                writer_texture_features = texture_model.adjust_nan_values(
                    np.reshape(writer_texture_features,
                               (texture_model.num_blocks_per_class, texture_model.get_num_features()))).tolist()

                try:
                    #print(base_samples_h+str(h_coeff)+"/Class"+str(class_number))
                    os.makedirs(base_samples_h+str(h_coeff)+"/Class"+str(class_number))
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise

                np.savetxt(base_samples_h+str(h_coeff)+"/Class"+str(class_number)+'/'+name, texture_features[0], delimiter=",")

                #create file
            print('Sift Model')

            for idx in range(0,5):
                print(idx)
                name = Path(filename).name
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
    print('TESTCASES - ',start)
    t = [1, 50, 150, 225, 300]
    phi = [36, 72, 108, 234, 360]
    h = [0.1, 0.3, 0.5, 0.7, 0.9]

    texture_model = TextureWriterIdentification()
    sift_model = SiftModel()
    sift_model.set_code_book('en')

    start_class = int(start)
    num_classes = int(num)

    base_path = 'C:/Users/omars/Documents/Github/LIWI/Omar/Dataset/Validation/testing'

    base_test_h = 'C:/Users/omars/Documents/Github/LIWI/Omar/Validation/TestCases/H/'
    base_test_t = 'C:/Users/omars/Documents/Github/LIWI/Omar/Validation/TestCases/SDS/'
    base_test_phi = 'C:/Users/omars/Documents/Github/LIWI/Omar/Validation/TestCases/SOH/'


    for class_number in range(start_class, num_classes + 1):


        writer_texture_features = []
        SDS_train = []
        SOH_train = []
        texture_model.num_blocks_per_class = 0
        print('Class' + str(class_number) + ':')

        # loop on training data for each writer
        for filename in glob.glob(
                base_path + str(class_number) + '_0.jpg'):
            print(filename)

            image = cv2.imread(filename)

            name = Path(filename).name
            name = name.replace('jpg','csv')
            print(name)
            print('Texture Features')
            # writer_texture_features.append(texture_model.get_features(cv2.imread(filename))[0].tolist())

            for h_coeff in h:
                print(h_coeff)
                _, texture_features = texture_model.get_features(image)
                writer_texture_features = np.append(writer_texture_features, texture_features[0].tolist())
                writer_texture_features = texture_model.adjust_nan_values(
                    np.reshape(writer_texture_features,
                               (texture_model.num_blocks_per_class, texture_model.get_num_features()))).tolist()

                np.savetxt(base_test_h+str(h_coeff)+'/'+name, texture_features[0], delimiter=",")

                #create file
            print('Sift Model')

            for idx in range(0,5):
                print(idx)
                SDS, SOH = sift_model.get_features(name, image=image,t=t[idx],phi=phi[idx])

                str_t = str(t[idx])
                str_phi = str(phi[idx])

                while len(str_t) < 3:
                    str_t = '0' + str_t

                while len(str_phi) < 3:
                    str_phi = '0' + str_phi

                np.savetxt(base_test_t+str_t+'/'+name, SDS, delimiter=",")
                np.savetxt(base_test_phi+str_phi+'/'+name, SOH, delimiter=",")





Samples_gen(58,59)
# for beg in range(300,350,20):
#     Testcase_gen(beg,20+beg)
#     Samples_gen(beg,20+beg)


