
from CaluculateNomral import ObjReadTif
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd


# def RawData(*number,Min = None, Max = None):
#     info = ObjReadSIF.nor_data
#     if number ==():
#      if (Min == None) and (Max == None):
#         print("All Data Selected")
#         Data = info
#      elif(Max != None):
#         print("Data Selected")
#         Data = info[:Max]
#      elif(Min != None) :
#         print("Data Selected")
#         Data = info[Min:]
#      else:
#         print("Data Selected")
#         Data = info[Min:Max]
#
#
#     else:
#         number = number[0]
#         print("Specific Data Selected, Picture NO.",number +1)
#         Data = info[number]
#     return Data
image = np.load('Datas/RobberPos.npy')
images = np.load('Datas/Robber.npy')

print(len(images))
#
def MaskedSample():

    # print("Image data before Normalize:\n", image)
    # print(image.dtype)
    normaliz = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8UC1)
    # print("Image data after Normalize:\n", normaliz)
    # print(normaliz.dtype)
    # plt.imshow(normaliz)
    # plt.gray()
    # plt.show()

    # bblr = cv.bilateralFilter(normaliz, 6, 6, 1000)
    mid = cv.medianBlur(normaliz, 15)

    egde = cv.Canny(mid, 0,200)

    contours,hierarchy = cv.findContours(egde,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # contour = contours[2]
    mask = np.zeros_like(normaliz)
    mask1 = cv.drawContours(mask,contours,-1,(255,255,255),-1)
    ##for five balls need to select the hierarchy
    mask2 = mask1.flatten()
    # print(mask2.shape)
    # print(mask2[231764])
    # # mask2 = cv.medianBlur(mask1, 11)
    out = cv.bitwise_and(normaliz,normaliz,mask=mask1)

    # plt.imshow(mask1,origin='lower')
    # plt.show()

    #
    # cv.imshow('egde',mask1)
    # # cv.imshow('bblr',mid)
    # # cv.imshow('normaliz', normaliz)
    #
    #
    # cv.waitKey(0)
    # cv.destroyAllWindows()


    output = []

    for i in range(len(images)):
     normalize = cv.normalize(images[i], None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8UC1)
     out1 = cv.bitwise_and(normalize, normalize, mask=mask1)
     out = out1/255
     out = out.flatten()
     # print(out.shape)
     # plt.imshow(out)
     # plt.gray()
     # plt.show()
     output.append(out)
    results = np.array(output)
    # print(results.shape)

    # b=[]
    # for i in range(150):
    #  Intensity = results[i].flatten()
    #  b.append(Intensity)
    #
    # Beach = np.array(b)
    # print("shape", Beach)

    return results,mask2,image,mask1

def Dirction_of_Lights():
    read = np.load('Datas/LEDs.npy')
    dirsoflights= np.array(read)
    return dirsoflights




if __name__ == '__main__':
    print(MaskedSample()[1])











