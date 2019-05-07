from texturemodel.texture_model import *
from horestmodel.horest_model import *
import pickle
from siftmodel.sift_model_test import *



code_book = pickle.load( open( "centers.pkl", "rb" ) )

#writer identification using SIFT
accuracy = None
print('start')
class_labels = list(range(2, 160))
classCombinations = list(combinations(class_labels, r=158))
total = len(classCombinations)
print(total)
count = 0
for x in classCombinations:
    print(x)
    sift_model = SiftModel(test_classes=x , code_book=code_book)
    sift_model.run()
    if accuracy is None:
        accuracy = sift_model.accuracy
    else:
        accuracy = np.append(accuracy,sift_model.accuracy,axis=1)
    print('Total accuracy',accuracy[0,count]/accuracy[1,count])
    np.savetxt("accuracyTester.csv", accuracy, delimiter=",")
    count +=1
    print(accuracy.shape)

