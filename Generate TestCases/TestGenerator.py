import xml.etree.ElementTree as ET
import glob
import cv2
from pathlib import Path
import os, errno
import numpy as np
import shutil
from itertools import combinations
import random
base = 'test/'


imageCount = np.zeros((700,1))
# for filename in glob.glob('iAm/*.xml'):
#     #temp = cv2.imread(filename)
#     tree = ET.parse(filename)
#     root = tree.getroot()
#     id = root.attrib[ 'writer-id']
#     imageCount[int(id)] += 1
#
#     filename = filename.replace('xml', 'png')
#     name = Path(filename).name
#     print(name)
#     try:
#         os.makedirs(base+id)
#     except OSError as e:
#         if e.errno != errno.EEXIST:
#             raise
#     shutil.copyfile(filename,base+id+'/'+name)
#
    #cv2.imwrite(base+id+'/'+name,temp)

base = 'Samples/'
try:
    os.makedirs('TestCases')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

# np.savetxt("foo.csv", imageCount, delimiter=",")
# imageCount = np.genfromtxt('foo.csv', delimiter=',')
classNum = 0
print('generating cases')
for i in range(0,1000):
    # if imageCount[i] < 3:
    #     continue
    classNum += 1
    id = str(i)
    print(i)
    try:
        os.makedirs(base+'Class'+str(classNum))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # while len(id) < 3:
    #     id = '0'+id

    while len(id) < 4:
        id = '0' + id

    count = 0
    for filename in glob.glob('KHATT/*'+id+'*.tif'):
        #temp = cv2.imread(filename)
        name = Path(filename).name
        if count < 3:
            #cv2.imwrite(base+'Class'+str(classNum)+'/'+name,temp)
            shutil.copyfile(filename, base+'Class'+str(classNum)+'/'+name)

        elif count == 3:
            #cv2.imwrite('TestCases/testing'+str(classNum)+'.png',temp)
            shutil.copyfile(filename, 'TestCases/testing'+str(classNum)+'.png')
            break
        count += 1

# try:
#     os.makedirs('exam')
# except OSError as e:
#     if e.errno != errno.EEXIST:
#         raise
#
# class_labels = list(range(1, 159))
# classCombinations = list(combinations(class_labels, r=3))
# total = len(classCombinations)
# for i in range(3, 6):
#     try:
#         os.makedirs('exam/'+str(i))
#     except OSError as e:
#         if e.errno != errno.EEXIST:
#             raise
#     for j in range(1,101):
#         try:
#             os.makedirs('exam/' + str(i)+'/'+str(j))
#         except OSError as e:
#             if e.errno != errno.EEXIST:
#                 raise
#         case = classCombinations[random.randint(0,total-1)]
#         print(case)
#         for k in range(1,4):
#             count = 1
#             for filename in glob.glob('Samples/Class' + str(case[k-1]) + '/*.png'):
#                 name = str(k)+str(count)+'.png'
#                 shutil.copyfile(filename, 'exam/' + str(i)+'/'+str(j)+ '/' + name)
#                 count += 1
#         classNum = random.randint(0,2)
#         print(classNum)
#         shutil.copyfile('TestCases/testing'+str(case[classNum])+'.png','exam/' + str(i)+'/'+str(j)+ '/test'+str(classNum+1)+'.png')
