from horestmodel.horest_model import *
import pickle
from siftmodel.sift_model_test import *

t = [1, 50, 150]
phi = [36, 72, 108]
w = [0.1,0.25,0.5,0.75,0.9]

t=[1]
phi=[36]
w=[0.75]
code_book = pickle.load( open( "centers_KHATT.pkl", "rb" ) )

#writer identification using SIFT

print('start')
class_labels = list(range(1, 171))
classCombinations = combinations(class_labels, r=170)
# total = len(classCombinations)
# print(total)
count = 0
acc = None
for radius in range (10,170,10):
    class_labels = list(range(1, 290))
    classCombinations = combinations(class_labels, r=radius)
    testcases = 0
    right_test_cases=0
    total_test_cases=0
    count = 0
    accuracy = None

    for x in classCombinations:
        print(testcases)
        for t_test in t:
            accuracy_10 = None
            for phi_test in phi:
                count=0
                for w_test in w:
                    # print('t ',t_test,' - phi ',phi_test,' - w ',w_test)
                    sift_model = SiftModel(test_classes=x , code_book=code_book,t=t_test,phi=phi_test,w=w_test)
                    sift_model.run()
                    # if accuracy is None:
                    #     accuracy = sift_model.accuracy
                    # else:
                    #     accuracy = np.append(accuracy,sift_model.accuracy,axis=1)
                    # print('Total accuracy',accuracy[0,count]/accuracy[1,count])
                    # np.savetxt("accuracyTester.csv", accuracy[0,:]/accuracy[1,:], delimiter=",")
                    #np.savetxt("varinacc.csv", sift_model.thesis, delimiter=",")
                    # count +=1
                    count += 1
                    testcases += sift_model.total_test_cases
                    right_test_cases += sift_model.right_test_cases
        if testcases > 5000:

            if acc is None:
                acc = np.array([radius, right_test_cases / testcases]).reshape((1, 2))
            else:
                acc = np.append(acc, np.array([radius, right_test_cases / testcases]).reshape((1, 2)),
                                axis=0)
            print('Acc finaal @', ' - ', right_test_cases / testcases, 'rad - ', radius)
            print('shape ', acc.shape)
            np.savetxt('ar_sift_validation.csv', acc, delimiter=',')
            break

                    # print(accuracy.shape)
                # accuracy = accuracy[0, :] / accuracy[1, :]
                # accuracy = accuracy.reshape((1,5))
                # if accuracy_10 is None:
                #     accuracy_10 = accuracy
                # else:
                #     accuracy_10 = np.append(accuracy_10, accuracy, axis=0)
                # print('accuracy_10 shape ',accuracy_10.shape)
                # np.savetxt("ar_t"+str(t_test)+'.csv',accuracy_10,delimiter=",")
                #







acc = None
for radius in range (10,170,10):
    class_labels = list(range(1, 290))
    classCombinations = combinations(class_labels, r=radius)
    testcases = 0
    right_test_cases=0
    total_test_cases=0
    count = 0
    accuracy = None

    for x in classCombinations:

        for t_test in t:
            for phi_test in phi:
                count=0
                for w_test in w:
                    # print('t ',t_test,' - phi ',phi_test,' - w ',w_test)
                    sift_model = SiftModel(test_classes=x, code_book=code_book, t=t_test, phi=phi_test, w=w_test)
                    sift_model.run()
                    # if accuracy is None:
                    #     accuracy = sift_model.accuracy
                    # else:
                        # accuracy = np.append(accuracy, sift_model.accuracy, axis=1)
                    # print('Total accuracy', accuracy[0, count]/accuracy[1, count])
                    # np.savetxt("accuracyTester.csv", accuracy[0, :]/accuracy[1, :], delimiter=",")
                    #np.savetxt("varinacc.csv", sift_model.thesis, delimiter=",")
                    count +=1
                    testcases += sift_model.total_test_cases
                    right_test_cases+= sift_model.right_test_cases
                    # print(accuracy.shape)
        if testcases > 5000:

            if acc is None:
                acc = np.array([radius , right_test_cases/testcases]).reshape((1, 2))
            else:
                acc = np.append(acc, np.array([radius, right_test_cases/testcases]).reshape((1, 2)), axis=0)
            print('Acc finaal @', ' - ', right_test_cases/testcases, 'rad - ', radius)
            print('shape ', acc.shape)
            np.savetxt('sift_validation.csv', acc, delimiter=',')
            break
