import numpy as np
import matplotlib.pyplot as plt


img=plt.imread("road.jpg")
img=img[:,:,0].copy()

plt.figure()
#a) posvijetliti sliku
plt.imshow(img,vmin=0,vmax=100,cmap="gray")
plt.show()
#b) prikazati samo drugu cetvrtinu slike po širini
plt.imshow(img[:, 160:320],cmap="gray")
plt.show()

#c) zarotirati sliku za 90 stupnjeva u smjeru kazaljke na satu
plt.imshow(np.rot90(img,3),cmap="gray")  #3 je broj koji označava koliko puta će se slika zarotirati a pošto ova funkcija rotira suprono od smjera kazaljke na satu
plt.show()                                        #da bismo dobili dobru poziciju moramo 3 puta zarotirati

#d) zrcaliti sliku
plt.imshow(np.fliplr(img),cmap="gray")
plt.show()

