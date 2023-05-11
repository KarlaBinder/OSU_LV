import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# train i test podaci
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# prikaz karakteristika train i test podataka
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

# TODO: prikazi nekoliko slika iz train skupa
for i in range(9):  
    plt.subplot(330 + 1 + i)
    plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
plt.show()

# skaliranje slike na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# slike trebaju biti (28, 28, 1)
x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

print("x_train shape:", x_train_s.shape)
print(x_train_s.shape[0], "train samples")
print(x_test_s.shape[0], "test samples")


# pretvori labele
y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)


# TODO: kreiraj model pomocu keras.Sequential(); prikazi njegovu strukturu
model = keras.Sequential()
model.add(layers.Input(shape=(784, )))
model.add(layers.Dense(100, activation="relu"))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))
model.summary()

# TODO: definiraj karakteristike procesa ucenja pomocu .compile()
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy",])


# TODO: provedi ucenje mreze
x_train = np.reshape(x_train_s, (len(x_train_s), x_train_s.shape[1]*x_train_s.shape[2]))        #dobiva se br slika puta broj piksela u jednoj slici
x_test = np.reshape(x_test, (len(x_test_s), x_test_s.shape[1]*x_test_s.shape[2]))

ohe = OneHotEncoder()
y_train_ohe = ohe.fit_transform(np.reshape(y_train, (-1, 1))).toarray()   #OneHotEncoder trazi 2d array, pa treba reshape (n,1)
y_test_ohe = ohe.fit_transform(np.reshape(y_test, (-1, 1))).toarray()

history = model.fit(x_train, y_train_ohe, batch_size=32, epochs=5, validation_split=0.1)
score = model.evaluate(x_test, y_test_ohe, verbose=0)


# TODO: Prikazi test accuracy i matricu zabune
y_test_pred = model.predict(x_test)      #ne vraca znamenke koje trebaju za confusion matrix,vraca za svaki primjer vektor vjerojatnosti pripadanja svakoj od 10 klasa (softmax) (10 000,10)
y_test_pred = np.argmax(y_test_pred, axis=1)  #vraća polje indeksa najvecih elemenata u svakom pojedinom retku (1d polju) (0-9) (10 000,) - 1d polje

cm = confusion_matrix(y_test, y_test_pred)
print("Matrica zabune:", cm)
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_pred))
disp.plot()
plt.show()

# TODO: spremi model
model.save('KerasModel')
