import cv2 as cv
import numpy as np
from skimage.filters import threshold_otsu
from skimage.filters import gaussian
from utils.filters import *

from PIL import Image

image = cv.imread('omar.png')
# print(image)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# temp = Image.open('test.png')
# temp = temp.convert('RGB')
print(gray)
print(np.max(gray))
threshold = threshold_otsu(gray)
        # threshold = filters.threhold_otsu(img)
gray[(gray > threshold)] = 255
gray[(gray <= threshold)] = 0

gray = 255 - gray
cv.imwrite('t.png',gray)

print(gray)
col = np.sum(gray/255,axis=1)
col = col.reshape((col.shape[0],1))
print(col.shape)
np.savetxt('line.csv',col,delimiter=',')