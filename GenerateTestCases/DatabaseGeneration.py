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
import copy


#start at 687,3240
#id starts at 672
def firemaker_preprocessing():
    base = 'C:/Users/omars/Documents/Github/LIWI/Omar/firemaker/firemaker/300dpi/'
    baseDB = 'C:/Users/omars/Documents/Github/LIWI/Omar/test/'
    id = 672
    id1 = '01.tif'
    id2 = '02.tif'
    id3 = '03.tif'
    id4 = '04.tif'
    file1 = 'p1-copy-normal'
    file2 = 'p2-copy-upper'
    file3 = 'p3-copy-forged'
    file4 = 'p4-self-natural'
    for filename in glob.glob(base +'*/*01.tif'):
        filename2 = copy.copy(filename)
        filename3 = copy.copy(filename)
        filename4 = copy.copy(filename)

        filename2 = filename2.replace(id1,id2)
        filename2 = filename2.replace(file1, file2)

        filename3 = filename3.replace(id1,id3)
        filename3 = filename3.replace(file1, file3)

        filename4 = filename4.replace(id1,id4)
        filename4 = filename4.replace(file1, file4)

        try:
            os.makedirs(baseDB+str(id)+'/')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        filename_arr = [filename,filename2,filename3,filename4]
        for item in filename_arr:
            temp = cv2.imread(item)
            temp = temp[687:3240, :]
            # temp = temp.convert('RGB')
            name = Path(item).name
            name = name.replace('.tif', '.jpg')
            cv2.imwrite(baseDB+str(id)+'/' + name,temp)

            # temp = Image.open(baseDB+str(id)+'/' + name)
            #
            # temp.save(baseDB+str(id)+'/' + name)

        print(filename)
        id += 1





# IAM
def test_generator():

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


    baseTraining = 'C:/Users/omars/Documents/Github/LIWI/Omar/Dataset/Training/'
    baseValidation = 'C:/Users/omars/Documents/Github/LIWI/Omar/Dataset/Validation/'
    baseTesting = 'C:/Users/omars/Documents/Github/LIWI/Omar/Dataset/Testing/'

    #
    # try:
    #     os.makedirs('C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/TestCasesCompressed/TestCases')
    # except OSError as e:
    #     if e.errno != errno.EEXIST:
    #         raise

    # np.savetxt("foo.csv", imageCount, delimiter=",")
    # imageCount = np.genfromtxt('foo.csv', delimiter=',')
    classNum = 0
    print('generating cases')
    for i in range(0,962):
        # if imageCount[i] < 3:
        #     continue
        classNum += 1
        id = str(i)
        print(i)
        try:
            os.makedirs(baseTraining+'Class'+str(classNum))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        while len(id) < 3:
            id = '0'+id
        count = 0
        for filename in glob.glob('C:/Users/omars/Documents/Github/LIWI/Omar/test/'+id+'/*.png'):
            # temp = cv2.imread(filename)
            temp = Image.open(filename)
            temp = temp.convert('RGB')
            name = Path(filename).name
            name = name.replace('.png', '.jpg')

            if count==0:
                #Training
                temp.save(baseTraining+'Class'+str(classNum)+'/'+name)
                # cv2.imwrite(base+'Class'+str(classNum)+'/'+name,temp)
                # shutil.copyfile(filename, base+'Class'+str(classNum)+'/'+name)

            elif count == 1:
                #Validation
                temp.save(baseValidation+'testing'+str(classNum)+'_'+str(count-1) + '.jpg')
                # cv2.imwrite('C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/TestCases/testing'+str(classNum)+'_'+str(count-1) + '.jpg',temp)
                # shutil.copyfile(filename, 'C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/TestCases/testing'+str(classNum)+'_'+str(count-1)+'.jpg')
            else:
                temp.save(baseTesting + 'testing' + str(classNum) + '_' + str(count - 1) + '.jpg')

            count += 1

        for filename in glob.glob('C:/Users/omars/Documents/Github/LIWI/Omar/test/'+id+'/*.jpg'):
            # temp = cv2.imread(filename)
            temp = Image.open(filename)
            name = Path(filename).name

            if count==0:
                #Training
                temp.save(baseTraining+'Class'+str(classNum)+'/'+name)
                # cv2.imwrite(base+'Class'+str(classNum)+'/'+name,temp)
                # shutil.copyfile(filename, base+'Class'+str(classNum)+'/'+name)

            elif count == 1:
                #Validation
                temp.save(baseValidation+'testing'+str(classNum)+'_'+str(count-1) + '.jpg')
                # cv2.imwrite('C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/TestCases/testing'+str(classNum)+'_'+str(count-1) + '.jpg',temp)
                # shutil.copyfile(filename, 'C:/Users/Samar Gamal/Documents/CCE/Faculty/Senior-2/2st term/GP/writer identification/LIWI/TestCases/testing'+str(classNum)+'_'+str(count-1)+'.jpg')
            else:
                temp.save(baseTesting + 'testing' + str(classNum) + '_' + str(count - 1) + '.jpg')

            count += 1



test_generator()

