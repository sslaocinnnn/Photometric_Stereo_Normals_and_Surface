import math

import numpy as np
import scipy as sp
from numpy.linalg import inv
from CaluculateNomral import Data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv
import sklearn.preprocessing as sk

position_of_light = Data.Dirction_of_Lights()
print(position_of_light)
alldata = Data.MaskedSample()
mask = alldata[1]
print('mask shape:',mask.shape)
pic = cv.normalize(alldata[2],None,0,255,cv.NORM_MINMAX,dtype=cv.CV_8UC1)

# cv.imshow('1',robberman[0])
# cv.waitKey(0)
# cv.destroyAllWindows()


data = alldata[0]
dirction = position_of_light

print('data shape',data.shape)
A = dirction
# print('A', A)

# print(Intensity[252410])

def lsqusenormalequation():

    solutions =[]
    albedos = []
    solution2 =[]
    for j in range(len(mask)):
        if mask[j] != 0:
         B = data[:, j,None]
         # print('B',B.shape)
         solution = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(B)
         # print('solution', solution.shape)
         unit = math.sqrt((solution[0,0]**2)+(solution[1,0]**2)+(solution[2,0]**2))
         solution = solution/unit
         albedo = math.sqrt((solution[0, 0] ** 2) + (solution[1, 0] ** 2) + (solution[2, 0] ** 2))
         print('abledo',albedo)
         normall = solution / albedo


         # solution2.append(solution)
         solutions.append(normall)
         albedos.append(albedo)



    normal = np.array(solutions).reshape(-1,3)
    # unnor = np.array(solution2).reshape(-1,3)

    # print('normal',normal.shape)


    x = np.array(normal[:, 0])
    y = np.array(normal[:, 1])
    z = np.array(normal[:, 2])

    recover = np.reshape(mask,(1040,1392))
    p = 0
    q = 0
    fig = plt.figure()
    plt.gray()
    ax = fig.add_subplot()

    for ii in range(0,1040):
        for jj in range(0,1392):
            if recover[ii,jj] !=0:
                X = (jj + x[q])
                Y = (ii + y[q])
                Z = (z[q])
                # ax.scatter(X,Y,Z)
                ax.plot([jj,X],[ii,Y])
                q = q+1

    ax.imshow(recover)

    plt.show()






    # normal = np.array(solutions)
    # alb = np.array(albedos)
    # alb = alb.flatten()
    # print(alb.shape)
    # picf = pic.flatten()
    #
    #
    #
    # n = 0
    # for i in range(len(picf)):
    #     if mask[i] !=0:
    #         picf[i] = picf[i]*alb[n]
    #         n = n+1
    #         print('n',n)
    #         print('i',i)
    # albmap = np.reshape(picf,[1040,1392])
    #
    # albmap = cv.normalize(albmap,None,0,255,cv.NORM_MINMAX,dtype=cv.CV_8UC1)
    # # print('values', albmap)
    #
    # cv.imshow('albmap',albmap)
    # cv.imshow('image',pic)
    #
    # cv.waitKey(0)
    # cv.destroyAllWindows()



if __name__ == '__main__':
    lsqusenormalequation()





