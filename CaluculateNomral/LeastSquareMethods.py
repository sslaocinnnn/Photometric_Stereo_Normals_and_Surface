

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
mask1 = alldata[3]
print('mask shape:',mask.shape)
np.save('Datas/Mask.npy', mask1)
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
    # solution2 =[]
    # for j in range(len(mask)):
    #      B = data[:, j]
    #      solution2.append(B)
    #
    # B = np.asarray(solution2).T
    # print('B',B.shape)

    solution = np.linalg.lstsq(A,data)[0].T
    print('solution', solution.shape)
    NN = np.reshape(solution,(1040,1392,3))


    normal = sk.normalize(solution, axis=1)

    N = np.reshape(normal,(1040,1392,3))
    np.save('Datas/Normal.npy', N)




         #
         # unit = math.sqrt((solution[0,0]**2)+(solution[1,0]**2)+(solution[2,0]**2))
         # normall= solution/unit
         # albedo = math.sqrt((solution[0, 0] ** 2) + (solution[1, 0] ** 2) + (solution[2, 0] ** 2))
         # print('abledo',albedo)
         # normall = solution / albedo


         # solution2.append(solution)
         #
         # solutions.append(solution)
         # # albedos.append(albedo)


    # normal = np.array(solutions).reshape(-1,3)
    print(normal)






    # # unnor = np.array(solution2).reshape(-1,3)
    # print(normal)
    #
    # # print('normal',normal.shape)
    #
    #
    # x = np.array(normal[:, 0])
    # y = np.array(normal[:, 1])
    # z = np.array(normal[:, 2])
    # #
    # recover = np.reshape(mask,(1040,1392))
    # #
    # q = 0
    # X=np.zeros([1040,1392])
    # Y=np.zeros([1040,1392])
    # Z=np.zeros([1040,1392])
    # N = np.zeros([1040,1392,3])
    # for ii in range(0,1040):
    #     for jj in range(0,1392):
    #         if recover[ii,jj] !=0:
    #             N[ii,jj,0]=x[q]
    #             N[ii, jj, 1] = y[q]
    #             N[ii, jj, 2] = z[q]

    #             X[ii,jj] = x[q]
    #             Y[ii,jj] = y[q]
    #             Z[ii,jj] = z[q]
    #             q = q+1
    #
    # X = np.array(X)
    # Y = np.array(Y)
    # Z = np.array(Z)
    # np.save('x.npy',X)
    # np.save('y.npy', Y)
    # np.save('z.npy', Z)

    N[:,:,0],N[:,:,2]= N[:,:,2],N[:,:,0].copy()
    N = (N+1.0)/2.0
    cv.imshow('normal',N)
    N = cv.convertScaleAbs(N, alpha=(255.0))
    cv.imwrite('Normal Map.jpg',N)
    cv.waitKey()
    cv.destroyAllWindows()







    # fig = plt.figure()
    #
    # plt.gray()
    # ax = fig.add_subplot()
    #
    # for ii in range(0,1040):
    #     for jj in range(0,1392):
    #         if recover[ii,jj] !=0:
    #             X = (jj + x[q])
    #             Y = (ii + y[q])
    #             # Z = (z[q])
    #             # ax.scatter(X,Y,Z)
    #             ax.plot([jj,X],[ii,Y],c='black')
    #             q = q+1
    #
    #
    #
    # plt.show()






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





