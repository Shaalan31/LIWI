from texturemodel.texture_model import *
from horestmodel.horest_model import *
import pickle
from siftmodel.sift_model import *

textureMethod = TextureWriterIdentification('D:/Uni/Graduation Project/All Test Cases/KHATT/Samples/Class',
                                            'D:/Uni/Graduation Project/All Test Cases/KHATT/TestCases/testing')
textureMethod.run()

#Horst Writer identification
# horestMethod = HorestWriterIdentification('D:/Uni/Graduation Project/All Test Cases/IAM/Samples/Class',
#                                             'D:/Uni/Graduation Project/All Test Cases/IAM/TestCases/testing')
# horestMethod.run()

#SAMAR
# code book
# code_book = pickle.load(open( "siftmodel/centers.pkl", "rb" ))
# sift_model = SiftModel(first_class=91, last_class=120, code_book=code_book)
# sift_model.run()

# sift_model.get_features("C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/Samples/Class75/f04-011.png","f04-011.png")
# sift_model.get_features("C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/Samples/Class114/g07-022b.png","g07-022b.png")
# sift_model.get_features("C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/Samples/Class112/g07-003b.png","g07-003b.png")
# sift_model.get_features("C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/TestCases/testing24_2.png","testing24_2.png")
# sift_model.get_features("C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/TestCases/testing48_3.png","testing48_3.png")
# sift_model.get_features("C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/TestCases/testing1_27.png","testing1_27.png")
# sift_model.get_features("C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/TestCases/testing1_9.png","testing1_9.png")
# sift_model.get_features("C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/KHATT/TestCases/testing1.png","testing1.png")
# sift_model.get_features("C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/KHATT/TestCases/testing290.png","testing290.png")
# sift_model.get_features("C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/KHATT/TestCases/testing798.png","testing798.png")
# sift_model.get_features("C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/KHATT/TestCases/testing187.png","testing187.png")

#End Samar

#Shaalan
# code_book = pickle.load( open( "siftmodel/centers.pkl", "rb" ) )
#
# #writer identification using SIFT
# accuracy = None
# for x in range(1,160,10):
#     sift_model = SiftModel(first_class=x , last_class=x+9, code_book=code_book)
#     sift_model.run()
#     if accuracy is None:
#         accuracy = sift_model.accuracy
#     else:
#         accuracy = np.append(accuracy,sift_model.accuracy,axis=1)
#     print('Total accuracy',accuracy)
#     print(accuracy.shape)
#
# np.savetxt("accuracy6.csv", accuracy, delimiter=",")
