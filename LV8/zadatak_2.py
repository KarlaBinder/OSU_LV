from keras.models import load_model
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#ucitavanje modela
model = load_model('KerasModel')
model.summary()

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

X_test_reshaped = np.reshape(X_test,(len(X_test),X_test.shape[1]*X_test.shape[2])) #od x_test 1D radimo 2D array kako bi mogli dalje raditi

#predikcija, za prikaz lose klasificiranih
y_predictions = model.predict(X_test_reshaped) 
y_predictions = np.argmax(y_predictions, axis=1)

#prikaz nekih krivih predikcija

errors = y_predictions[y_predictions != y_test]   #krive predikcije modela
correct = y_test[y_predictions != y_test]  #ispravke krivih predikcija (koje je model promasio i stavio krive)
image_error = X_test[y_predictions != y_test]     
fig, axs = plt.subplots(2,3, figsize=(12,9))
br=0 
for i in range(2):
    for j in range(3):
        axs[i,j].imshow(image_error[br])
        axs[i,j].set_title(f'Model predvidio {errors[br]}, zapravo je {correct[br]}')
        br=br+1
plt.show()

