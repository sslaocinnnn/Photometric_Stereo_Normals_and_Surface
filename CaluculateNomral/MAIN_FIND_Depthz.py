import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy as sp
import cv2 as cv

N = np.load('../Datas/N.npy')

mask = np.load('../Datas/mask.npy')


index_y, index_x = np.where(mask!=0)
zlocal = np.zeros_like(mask)

numpixels = np.size(index_y)


A = sp.sparse.lil_matrix((2*numpixels,numpixels))
B = np.zeros((2*numpixels,1))


for ii in range(np.size(index_y)):
    zlocal[index_y[ii],index_x[ii]]=np.array(ii).astype(int)

for idx in range(numpixels):
  yy = index_y[idx]
  xx = index_x[idx]

  n_x = N[yy,xx,0]
  n_y = N[yy,xx,1]
  n_z = N[yy,xx,2]



  yidx = 2*idx

  if mask[yy,xx+1]:
      A[yidx,idx] = -1
      xidx = zlocal[yy,xx+1]
      A[yidx,xidx] = 1
      B[yidx] = -(n_x/n_z)
  elif mask[yy,xx-1]:
      xidx = zlocal[yy,xx-1]
      A[yidx,xidx]=-1
      A[yidx,idx]=1
      B[yidx] = -(n_x/n_z)

  y_idx = 2*idx +1

  if mask[yy+1,xx]:
      A[y_idx,idx] = 1
      xidx = zlocal[yy+1,xx]
      A[y_idx,xidx] = -1
      B[y_idx] = -(n_y/n_z)
  elif mask[yy-1,xx]:
      xidx = zlocal[yy-1,xx]
      A[y_idx,xidx]=1
      A[y_idx,idx]=-1
      B[y_idx] = -(n_y/n_z)

AtA = A.T.dot(A)
AtB = A.T.dot(B)

dep = sp.sparse.linalg.cg(AtA,AtB)[0]
print(dep)

#####clear the errors https://blog.csdn.net/SZU_Kwong/article/details/112757354
std_z = np.std(dep,ddof=1)
mean_z = np.mean(dep)
z_zscore = (dep-mean_z)/(std_z)

outlier_ind = np.abs(z_zscore)>10
z_min = np.min(dep[~outlier_ind])
z_max = np.max(dep[~outlier_ind])

Z = np.zeros_like(mask)


for j in range(np.size(index_y)):
    h = index_y[j]
    w = index_x[j]
    Z[h,w]=(dep[j]-z_min)/(z_max-z_min)*255
    # Z[h, w] = dep[j]*255


# cv.imshow('z',Z)
# cv.waitKey()
# cv.destroyAllWindows()
plt.imshow(Z)
plt.gray()

plt.title('Depth Map')
plt.savefig('Depth Map')
plt.show()

np.save('DepthMap.npy',Z)
