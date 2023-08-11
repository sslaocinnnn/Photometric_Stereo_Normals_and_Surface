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

# cv.imshow('1',robberman[0])
# cv.waitKey(0)
# cv.destroyAllWindows()


data = alldata[0]
dirction = position_of_light

print(data.shape)
A = dirction
# print(Intensity[252410])

def normals():


    solutions =[]
    albedos = []
    for j in range(data.shape[1]):

        if mask[j]!= 0.0:
         B = data[:, j]
         solution = np.linalg.inv(np.dot(A.T, A)).dot(A.T).dot(B)
         albedo = np.linalg.norm(solution)
         print(albedo)
         normall = solution / albedo
         value = normall
         solutions.append(value)
         albedos.append(albedo)


    normal = np.array(solutions)
    alb = np.array(albedos)
    alb = alb.flatten()
    print(alb.shape)


    albmap = np.zeros_like(mask)
    n = 0
    for i in range(len(mask)):
        if mask[i] !=0:
            albmap[i] = alb[n]
            n = n+1
            print('n',n)
            print('i',i)
    albmap = np.reshape(albmap,[1040,1392])

    albmap = cv.normalize(albmap,None,0,255,cv.NORM_MINMAX,dtype=cv.CV_8UC1)
    print('values', albmap)

    plt.gray()
    plt.imshow(albmap)
    plt.show()





    # print(normal)
    #
    # x = np.array(normal[:, 0])
    # y = np.array(normal[:, 1])
    # z = np.array(normal[:, 2])
    # fullimage=[]
    # fulimgae =[]
    # j =0
    # for i in range(mask.shape[0]):
    #     if mask[i] != 0:
    #      full = x[j],y[j],z[j]
    #      j = j+1
    #     else:full = np.array([0,0,0])
    #     fullimage.append(full)
    # fulimgae = np.array(fullimage)
    # print(fulimgae)

    # X = np.array(fulimgae[:, 0])
    # Y = np.array(fulimgae[:, 1])
    # Z = np.array(fulimgae[:, 2])

    # X = X.reshape(1,X.shape[0])
    # Y = Y.reshape(1,Y.shape[0] )
    # Z = Z.reshape(1,Z.shape[0])
    #
    # print(X.shape)
    # print(Y.shape)
    # print(Z.shape)









    # for i in range(len(Bs)):

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    #
    # ax.plot_trisurf(x,y,z)
    # ax.scatter(0, 0, 1, c='red')
    # # ax.scatter(X, Y, Z, c='yellow')
    # ax.view_init(0,0)
    # plt.show()



    #solution = np.linalg.lstsq(A,B)
    #solution = np.linalg.inv(np.dot(A.T, A)).dot(A.T).dot(B)

    # print("solution:",solution)

   #need find inverse of non-square matrix
   # print(Q.shape)
   # print(R.shape)
   # print(Q)
   # print(R)



if __name__ == '__main__':
    normals()



