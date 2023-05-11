import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

#1.Koliko je različitih boja prisutno u ovoj slici?
print(f'Broj pojedinačnih boja u slici: {len(np.unique(img_array_aprox, axis=0))}')

#2.i 3. Primijenite algoritam K srednjih vrijednosti koji ce pronaci grupe u RGB vrijednostima elemenata originalne slike.
#Vrijednost svakog elementa slike originalne slike zamijeni s njemu pripadaju´cim centrom.

#6. trazenje optimalnog broja grupa (K) 

error = []
for i in range (1,20):
    kmeans = KMeans(n_clusters=i,n_init=10, init='random')
    kmeans.fit(img_array_aprox)
    error.append(kmeans.inertia_)

plt.plot(range(1,20), error)
plt.xlabel('K')
plt.ylabel('J')
plt.show()

#Po grafu zaključujemo da je optimalni broj K=4

#Primjerna algoritma K srednjih vrijednosti

kmeans = KMeans(n_clusters=4, random_state=0).fit(img_array_aprox)
labels = kmeans.predict(img_array_aprox)
#zamjena originalne slike sa njemu pripadajućim centrom
img_array_quantized = kmeans.cluster_centers_[labels]
img_quantized = np.reshape(img_array_quantized, (w, h, d))
img_quantized = (img_quantized * 255).astype(np.uint8)  #pikseli su u floatu!
plt.imshow(img_quantized)
plt.title("Kvantizirana slika")
plt.show()
"""
km = KMeans(n_clusters=4, init='random', n_init=5, random_state=0)
lable = km.fit_predict(img_array_aprox)
rgb_cols = km.cluster_centers_.astype(np.float64)
img_quant = np.reshape(rgb_cols[lable], (w,h,d))

plt.figure()
plt.imshow(img_quant)
plt.show()
"""
#7.Elemente slike koji pripadaju jednoj grupi prikažite kao zasebnu binarnu sliku

for i in range(4) :
        binar = labels ==[i]
        new_image = np.reshape(binar, (img.shape[0:2]))
        new_image = new_image*1
        x=int(i/2)
        y=i%2
        plt.imshow(new_image,cmap='gray')
        plt.show()
