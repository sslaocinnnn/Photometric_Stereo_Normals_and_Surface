# import sif_parser
# import numpy as np
# from numpy import unravel_index
# import matplotlib.pyplot as plt
# import pandas as pd
# import xarray
# import xarray as xr
# import cv2 as cv
# import os
#
# imag_data = []
# nor_data = []
# def readsif(path):
#     for filename in os.listdir(path):
#         print(filename)
#         data = xr.DataArray(sif_parser.xr_open(path + '/' + filename))
#         twodims = data.isel(Time = 0)
#         TwoD = twodims.values
#         print(TwoD)
#         Nor = cv.normalize(TwoD, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8UC1)
#         print(Nor)
#         # plt.imshow(TwoD)
#         # plt.title(filename)
#         # plt.show()
#         imag_data.append(TwoD)
#         nor_data.append(Nor)
#
#
#
#
#
# # def BrightestPoint(imag):
# #  Imag = imag_data[imag]
# #  index = unravel_index(Imag.argmax(),Imag.shape)
# #  return index
# #
# #
#
#

#
#
#
#
#
#
#
#
#
#
# #
# # da1 = da/da.max()
# # #
# # # c = np.argmax(da1)
# # # # print(c)
# # d = unravel_index(da1.argmax(),da1.shape)
# # print(d)
# # print(da1[d])
#
#
#
#
#
#
# #
# # plt.imshow(da1)
# #
# # plt.show()
#
#
#
#
#
#
# # cv.imshow('da',da)
# # cv.waitKey(0)
# #
# # # closing all open windows
# # cv.destroyAllWindows()
import tifffile
objimage = []
# def objreadtif(pathofrobber= '',pathofpositions = '',Shape =None):
def objreadtif(
            pathofrobber='',
            pathofpositions='',
            Shape=None):
    read_image = tifffile.imread(pathofrobber)
    read_positions = tifffile.imread(pathofpositions)


    for i in range(150):
        print(i)
        read = read_image[i,:,:]
        objimage.append(read)
    if Shape == None:
        return objimage
    else:
        return read_positions











if __name__ == '__main__':
    objreadtif()









