from texturemodel.texture_model import *
from horestmodel.horest_model import *
import pickle
from siftmodel.sift_model import *

# textureMethod = TextureWriterIdentification('D:/Uni/Graduation Project/All Test Cases/KHATT/Samples/Class',
#                                             'D:/Uni/Graduation Project/All Test Cases/KHATT/TestCases/testing')
# textureMethod.run()

#Horst Writer identification
# horestMethod = HorestWriterIdentification('D:/Uni/Graduation Project/All Test Cases/IAM/Samples/Class',
#                                             'D:/Uni/Graduation Project/All Test Cases/IAM/TestCases/testing')
# horestMethod.run()

# code book
code_book = pickle.load( open( "siftmodel/centers.pkl", "rb" ) )

#writer identification using SIFT
accuracy = None
for x in range(1,160,10):
    sift_model = SiftModel(first_class=x , last_class=x+9, code_book=code_book)
    sift_model.run()
    if accuracy is None:
        accuracy = sift_model.accuracy
    else:
        accuracy = np.append(accuracy,sift_model.accuracy,axis=1)
    print('Total accuracy',accuracy)
    print(accuracy.shape)

np.savetxt("accuracy6.csv", accuracy, delimiter=",")