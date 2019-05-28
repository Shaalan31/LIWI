from texturemodel.texture_model import *
from siftmodel.sift_model import *
import os, errno
from server.utils.utilities import *



def Samples_gen(start,num):
    print('SAMPPLESS - ',start)

    h = [0.1, 0.3, 0.5, 0.7, 0.9]
    # h=[0.5,0.7]
    texture_model = TextureWriterIdentification()

    start_class = int(start)
    num_classes = int(num)

    base_path = 'C:/Users/omars/Documents/Github/LIWI/Omar/KHATT/Samples/Class'
    base_samples_h = 'C:/Users/omars/Documents/Github/LIWI/Omar/ValidationArabic/Samples/H/'



    for class_number in range(start_class, num_classes + 1):


        writer_texture_features = []
        SDS_train = []
        SOH_train = []
        texture_model.num_blocks_per_class = 0
        print('Class' + str(class_number) + ':')

        # loop on training data for each writer
        for filename in glob.glob(
                base_path + str(class_number) + '/*.tif'):
            print(filename)

            image = cv2.imread(filename)

            name = Path(filename).name
            name = name.replace('tif','csv')
            print('Texture Features')
            # writer_texture_features.append(texture_model.get_features(cv2.imread(filename))[0].tolist())

            for h_coeff in h:
                print(h_coeff)
                _, texture_features = texture_model.get_features(image)
                writer_texture_features = np.append(writer_texture_features, texture_features[0].tolist())
                writer_texture_features = texture_model.adjust_nan_values(
                    np.reshape(writer_texture_features,
                               (texture_model.num_blocks_per_class, texture_model.get_num_features())))

                try:
                    #print(base_samples_h+str(h_coeff)+"/Class"+str(class_number))
                    os.makedirs(base_samples_h+str(h_coeff)+"/Class"+str(class_number))
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise

                np.savetxt(base_samples_h+str(h_coeff)+"/Class"+str(class_number)+'/'+name, writer_texture_features, delimiter=",")








def Testcase_gen(start,num):
    print('TESTCASES - ',start)
    h = [0.1, 0.3, 0.5, 0.7, 0.9]

    # h = [ 0.5, 0.7]

    texture_model = TextureWriterIdentification()

    start_class = int(start)
    num_classes = int(num)

    base_path = 'C:/Users/omars/Documents/Github/LIWI/Omar/KHATT/TestCases/'

    base_test_h = 'C:/Users/omars/Documents/Github/LIWI/Omar/ValidationArabic/TestCases/H/'

    for class_number in range(start_class, num_classes + 1):


        writer_texture_features = []
        SDS_train = []
        SOH_train = []
        texture_model.num_blocks_per_class = 0
        print('Class' + str(class_number) + ':')

        # loop on training data for each writer
        for filename in glob.glob(
                base_path + 'testing'+str(class_number) + '.png'):


            image = cv2.imread(filename)

            name = Path(filename).name
            name = name.replace('png','csv')
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

                try:
                    # print(base_samples_h+str(h_coeff)+"/Class"+str(class_number))
                    os.makedirs(base_test_h+str(h_coeff))
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise

                np.savetxt(base_test_h+str(h_coeff)+'/'+name, writer_texture_features, delimiter=",")

                #create file





# Samples_gen(1,19)

for beg in range(121,300,20):
    Testcase_gen(beg,20+beg)
    Samples_gen(beg,20+beg)

#
# for beg in range(124, 350, 20):
#     adpp.Testcase_gen(beg, 20 + beg)
#     adpp.Samples_gen(beg, 20 + beg)