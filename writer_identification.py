import pickle
from siftmodel.sift_model import *

# code book
code_book = pickle.load(open( "siftmodel/centers.pkl", "rb" ))
sift_model = SiftModel(first_class=1, last_class=159, code_book=code_book)
sift_model.run()

# sift_model.get_features("C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/Samples/Class75/f04-011.png","f04-011.png")
# sift_model.get_features("C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/Samples/Class114/g07-022b.png","g07-022b.png")
# sift_model.get_features("C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/Samples/Class112/g07-003b.png","g07-003b.png")
# sift_model.get_features("C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/TestCases/testing24_2.png","testing24_2.png")
# sift_model.get_features("C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/TestCases/testing48_3.png","testing48_3.png")
# sift_model.get_features("C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/TestCases/testing1_27.png","testing1_27.png")
# sift_model.get_features("C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/TestCases/testing1_9.png","testing1_9.png")
