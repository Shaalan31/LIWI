import pickle
from siftmodel.sift_model import *

# code book
code_book = pickle.load( open( "siftmodel/centers.pkl", "rb" ) )
sift_model = SiftModel(first_class=101 , last_class=110, code_book=code_book)
sift_model.run()
