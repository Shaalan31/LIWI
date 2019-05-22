import xml.etree.ElementTree as ET
import glob
import cv2
from pathlib import Path
import os, errno
import numpy as np
import shutil
from itertools import combinations
import random
from PIL import Image


# KHATT
def khatt_test_generator():
    base = 'D:/Uni/Graduation Project/Samples/'
    try:
        os.makedirs('D:/Uni/Graduation Project/TestCases')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # np.savetxt("foo.csv", imageCount, delimiter=",")
    # imageCount = np.genfromtxt('foo.csv', delimiter=',')
    classNum = 0
    print('generating cases')
    for i in range(0, 1182):
        # if imageCount[i] < 3:
        #     continue
        classNum += 1
        id = str(i)
        print(i)
        try:
            os.makedirs(base + 'Class' + str(classNum))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        # while len(id) < 3:
        #     id = '0'+id

        while len(id) < 4:
            id = '0' + id

        count = 0
        for filename in glob.glob('D:/Uni/Graduation Project/KHATT/KHATT/*' + id + '*.tif'):
            # temp = cv2.imread(filename)
            name = Path(filename).name
            if count < 3:
                # cv2.imwrite(base+'Class'+str(classNum)+'/'+name,temp)
                shutil.copyfile(filename, base + 'Class' + str(classNum) + '/' + name)

            elif count == 3:
                # cv2.imwrite('TestCases/testing'+str(classNum)+'.png',temp)
                shutil.copyfile(filename, 'D:/Uni/Graduation Project/TestCases/testing' + str(classNum) + '.png')
                break
            count += 1
        if count == 0:
            classNum -= 1


# IAM
def iam_test_generator():
    # base = 'C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/test/'
    # imageCount = np.zeros((700,1))
    # for filename in glob.glob('C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/iAm/*.xml'):
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
    #     # cv2.imwrite(base+id+'/'+name,temp)

    base = 'C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/TestCasesCompressed/Samples/'
    try:
        os.makedirs(
            'C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/TestCasesCompressed/TestCases')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # np.savetxt("foo.csv", imageCount, delimiter=",")
    imageCount = np.genfromtxt('foo.csv', delimiter=',')
    classNum = 0
    print('generating cases')
    for i in range(0, 700):
        if imageCount[i] < 3:
            continue
        classNum += 1
        id = str(i)
        print(i)
        try:
            os.makedirs(base + 'Class' + str(classNum))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        while len(id) < 3:
            id = '0' + id
        count = 0
        for filename in glob.glob(
                'C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/test/' + id + '/*.png'):
            # temp = cv2.imread(filename)
            temp = Image.open(filename)
            temp = temp.convert('RGB')
            name = Path(filename).name

            if count < 2:
                name = name.replace('.png', '.jpg')
                temp.save(base + 'Class' + str(classNum) + '/' + name)
                # cv2.imwrite(base+'Class'+str(classNum)+'/'+name,temp)
                # shutil.copyfile(filename, base+'Class'+str(classNum)+'/'+name)

            elif count >= 2:
                temp.save(
                    'C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/TestCasesCompressed/TestCases/testing' + str(
                        classNum) + '_' + str(count - 1) + '.jpg')
                # cv2.imwrite('C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/TestCases/testing'+str(classNum)+'_'+str(count-1) + '.jpg',temp)
                # shutil.copyfile(filename, 'C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/TestCases/testing'+str(classNum)+'_'+str(count-1)+'.jpg')

            count += 1


# CEDAR
def cedar_test_generator():
    base = 'D:/Uni/Graduation Project/All Test Cases/CEDAR/Training/'
    try:
        os.makedirs('D:/Uni/Graduation Project/All Test Cases/CEDAR/Testing')
        os.makedirs('D:/Uni/Graduation Project/All Test Cases/CEDAR/Validation')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    classNum = 962
    print('generating cases')
    for i in range(1, 501):

        classNum += 1
        id = str(i)
        print(i)
        try:
            os.makedirs(base + 'Class' + str(classNum))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        while len(id) < 4:
            id = '0' + id
        count = 0
        for filename in glob.glob(
                'D:/Uni/Graduation Project/All Test Cases/CEDAR-Letter-FULL/png_clet_0001-500/' + id + '*.png'):
            # temp = cv2.imread(filename)
            temp = Image.open(filename)
            temp = temp.convert('RGB')
            name = Path(filename).name

            if count == 0:
                temp.save(
                    'D:/Uni/Graduation Project/All Test Cases/CEDAR/Validation/testing' + str(classNum) + '_' + str(
                        count) + '.jpg')


            elif count == 1:
                temp.save('D:/Uni/Graduation Project/All Test Cases/CEDAR/Testing/testing' + str(classNum) + '_' + str(
                    count) + '.jpg')

            elif count > 1:
                name = name.replace('.png', '.jpg')
                temp.save(base + 'Class' + str(classNum) + '/' + name)
            count += 1


cedar_test_generator()
