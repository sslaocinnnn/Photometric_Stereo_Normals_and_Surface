import Data
import matplotlib.pyplot as plt

x= Data.RawData(Min = 1)
print(x)
for i in range(5):
    plt.imshow(x[i])
    plt.show()
