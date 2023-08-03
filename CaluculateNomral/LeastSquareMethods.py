import numpy as np
import scipy as sp
from numpy.linalg import inv
from CaluculateNomral import Data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv

position_of_light = Data.Dirction_of_Lights()
print(position_of_light)
alldata = Data.MaskedSample()
mask = alldata[1]
print('mask shape:',mask.shape)

# cv.imshow('1',robberman[0])
# cv.waitKey(0)
# cv.destroyAllWindows()


data = alldata[0]
dirction = position_of_light

print('data shape',data.shape)
A = dirction
print('A', A)

# print(Intensity[252410])

def lsqusenormalequation():

    solutions =[]
    albedos = []
    for j in range(len(mask)):
        if mask[j] != 0:
         B = data[:, j,None]
         # print('B',B.shape)
         solution = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(B)
         # print('solution', solution.shape)
         albedo = np.linalg.norm(solution)
         normall = solution / albedo
         value = normall
         solutions.append(value)
         albedos.append(albedo)



    normal = np.array(solutions).reshape(-1,3)
    print(normal.shape)
    # print('normal',normal.shape)

    x = np.array(normal[:, 0])*10
    y = np.array(normal[:, 1])*10
    z = np.array(normal[:, 2])

    recover = np.reshape(mask,(1040,1392))
    p = 0
    q = 0
    fig = plt.figure()
    ax = fig.add_subplot()

    for ii in range(0,1040,2):
        for jj in range(0,1392,2):
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
    #
    #
    # albmap = np.zeros_like(mask)
    # n = 0
    # for i in range(len(mask)):
    #     if mask[i] !=0:
    #         albmap[i] = alb[n]
    #         n = n+1
    #         print('n',n)
    #         print('i',i)
    # albmap = np.reshape(albmap,[1040,1392])
    #
    # albmap = cv.normalize(albmap,None,0,255,cv.NORM_MINMAX,dtype=cv.CV_8UC1)
    # print('values', albmap)
    #
    # plt.gray()
    # plt.imshow(albmap)
    # plt.show()


if __name__ == '__main__':
    lsqusenormalequation()





