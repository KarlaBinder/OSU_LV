import numpy as np
import matplotlib.pyplot as plt

img=plt.imread("road.jpg")
img=img[:,:,0].copy()

plt.figure()

plt.imshow(img,vmin=0,vmax=100,cmap="gray")
plt.show()

plt.imshow(img,cmap="gray")
plt.show()

plt.imshow(np.rot90(img,3),cmap="gray")
plt.show()

plt.imshow(np.fliplr(img),cmap="gray")
plt.show()

plt.imshow(img[:, 160:320],cmap="gray")
plt.show()