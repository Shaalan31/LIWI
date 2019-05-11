from siftmodel.codebook import *


SOFM = Som_KHATT()
# SOFM.generate_codebook()
for x in range(10):
    print(x)
    SOFM.train_loop()