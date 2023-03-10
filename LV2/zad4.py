import numpy as np
import matplotlib.pyplot as plt

x=np.zeros((50,50))
y=np.ones((50,50))

first=np.vstack((x,y))
second=np.vstack((y,x))

img=np.hstack((first,second))
plt.imshow(img,cmap="gray")
plt.show()