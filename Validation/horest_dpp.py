from siftmodel.sift_model import *
from horestmodel.horest_model import *
import os, errno
from server.utils.utilities import *



def Samples_gen(start,num):
    print('SAMPPLESS - ',start)

    # h=[0.5,0.7]
    horest_model = HorestWriterIdentification()

    start_class = int(start)
    num_classes = int(num)

    base_path = 'D:/Uni/Graduation Project/All Test Cases/Dataset/Training/Class'
    base_samples_horest_csv = 'D:/Uni/Graduation Project/All Test Cases/Dataset/Validation/Samples/Horest/'



    for class_number in range(start_class, num_classes + 1):


        writer_horest_features = []
        horest_model.num_lines_per_class = 0
        print('Class' + str(class_number) + ':')

        # loop on training data for each writer
        for filename in glob.glob(
                base_path + str(class_number) + '/*.jpg'):
            print(filename)

            image = cv2.imread(filename)

            name = Path(filename).name
            name = name.replace('jpg','csv')
            print('Horest Features')
            # writer_texture_features.append(texture_model.get_features(cv2.imread(filename))[0].tolist())

            _, horest_features = horest_model.get_features(image)
            writer_horest_features = np.append(writer_horest_features, horest_features[0].tolist())
            writer_horest_features = horest_model.adjust_nan_values(
                np.reshape(writer_horest_features,
                           (horest_model.num_lines_per_class, horest_model.get_num_features())))

            try:
                #print(base_samples_h+str(h_coeff)+"/Class"+str(class_number))
                os.makedirs(base_samples_horest_csv+"/Class"+str(class_number))
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

            np.savetxt(base_samples_horest_csv+"/Class"+str(class_number)+'/'+name, writer_horest_features, delimiter=",")

def testing_gen(start,num):
    print('SAMPPLESS - ',start)

    # h=[0.5,0.7]
    horest_model = HorestWriterIdentification()

    start_class = int(start)
    num_classes = int(num)

    base_path = 'D:/Uni/Graduation Project/All Test Cases/Dataset/Validation/testing'
    base_samples_horest_csv = 'D:/Uni/Graduation Project/All Test Cases/Dataset/Validation/TestCases/Horest/'



    for class_number in range(start_class, num_classes + 1):


        writer_horest_features = []
        horest_model.num_lines_per_class = 0
        print('Class' + str(class_number) + ':')

        # loop on training data for each writer
        for filename in glob.glob(
                base_path + str(class_number) + '_0.jpg'):
            print(filename)

            image = cv2.imread(filename)

            name = Path(filename).name
            name = name.replace('jpg','csv')
            print('Horest Features')
            # writer_texture_features.append(texture_model.get_features(cv2.imread(filename))[0].tolist())

            _, horest_features = horest_model.get_features(image)
            writer_horest_features = np.append(writer_horest_features, horest_features[0].tolist())
            writer_horest_features = horest_model.adjust_nan_values(
                np.reshape(writer_horest_features,
                           (horest_model.num_lines_per_class, horest_model.get_num_features())))

            try:
                #print(base_samples_h+str(h_coeff)+"/Class"+str(class_number))
                os.makedirs(base_samples_horest_csv)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

            np.savetxt(base_samples_horest_csv+'/'+name, writer_horest_features, delimiter=",")






for beg in range(1,300,20):
    testing_gen(beg,20+beg)
    Samples_gen(beg,20+beg)

#
# for beg in range(124, 350, 20):
#     adpp.Testcase_gen(beg, 20 + beg)
#     adpp.Samples_gen(beg, 20 + beg)