from texturemodel.texture_model import *
from horestmodel.horest_model import *
import pickle
from siftmodel.sift_model import *

# textureMethod = TextureWriterIdentification('D:/Uni/Graduation Project/All Test Cases/KHATT/Samples/Class',
#                                             'D:/Uni/Graduation Project/All Test Cases/KHATT/TestCases/testing')
# textureMethod.run()

#Horst Writer identification
horestMethod = HorestWriterIdentification('D:/Uni/Graduation Project/All Test Cases/IAM/Samples/Class',
                                            'D:/Uni/Graduation Project/All Test Cases/IAM/TestCases/testing')
horestMethod.run()

# code book
code_book = pickle.load( open( "siftmodel/centers.pkl", "rb" ) )

#writer identification using SIFT
sift_model = SiftModel(first_class=101 , last_class=110, code_book=code_book)
sift_model.run()
