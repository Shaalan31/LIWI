from horestmodel.horest_model import *
import pickle
from siftmodel.sift_model_test import *

t = [1, 50, 150]
phi = [36, 72, 108]
w = [0.1,0.25,0.5,0.75,0.9]


code_book = pickle.load( open( "centers_KHATT.pkl", "rb" ) )

#writer identification using SIFT

print('start')
class_labels = list(range(1, 161))
classCombinations = combinations(class_labels, r=160)
# total = len(classCombinations)
# print(total)
count = 0
for x in classCombinations:
    for t_test in t:
        accuracy_10 = None
        for phi_test in phi:
            accuracy = None
            count=0
            for w_test in w:
                print('t ',t_test,' - phi ',phi_test,' - w ',w_test)
                sift_model = SiftModel(test_classes=x , code_book=code_book,t=t_test,phi=phi_test,w=w_test)
                sift_model.run()
                if accuracy is None:
                    accuracy = sift_model.accuracy
                else:
                    accuracy = np.append(accuracy,sift_model.accuracy,axis=1)
                print('Total accuracy',accuracy[0,count]/accuracy[1,count])
                np.savetxt("accuracyTester.csv", accuracy[0,:]/accuracy[1,:], delimiter=",")
                #np.savetxt("varinacc.csv", sift_model.thesis, delimiter=",")
                count +=1
                print(accuracy.shape)
            accuracy = accuracy[0, :] / accuracy[1, :]
            accuracy = accuracy.reshape((1,5))
            if accuracy_10 is None:
                accuracy_10 = accuracy
            else:
                accuracy_10 = np.append(accuracy_10, accuracy, axis=0)
            print('accuracy_10 shape ',accuracy_10.shape)
            np.savetxt("t"+str(t_test)+'p'+str(phi_test)+'.csv',accuracy_10,delimiter=",")

