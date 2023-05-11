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

# Perform KMeans clustering on the pixel values
k_values = range(1, 11)
inertias = []
labels_list = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0).fit(img_array)
    labels = kmeans.predict(img_array)
    img_array_aprox = kmeans.cluster_centers_[labels]
    img_aprox = np.reshape(img_array_aprox, (w, h, d))
    inertias.append(kmeans.inertia_)
    labels_list.append(labels)
"""
    # Show the image approximation for the current value of K
    plt.figure()
    plt.title(f"K={k}")
    plt.imshow(img_aprox)
    plt.tight_layout()
    plt.show()
"""
"""
# Plot the inertia vs. K curve
plt.figure()
plt.plot(k_values, inertias, 'bx-')
plt.xlabel('K')
plt.ylabel('Inertia')
plt.title('Inertia vs. K')
plt.show()
"""
# Show the binary images for each cluster
for k in k_values:
    labels = labels_list[k-1]
    for i in range(k):
        cluster_mask = np.zeros((w*h,), dtype=bool)
        cluster_mask[labels == i] = True
        cluster_img = np.reshape(cluster_mask, (w, h))
        plt.figure()
        plt.title(f"K={k}, Cluster {i}")
        plt.imshow(cluster_img, cmap='gray')
        plt.tight_layout()
        plt.show()
