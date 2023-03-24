import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import sklearn.linear_model as lm
import sklearn.metrics as skmetrics

input = pd.read_csv("data_C02_emission.csv")
X_labels = ['Fuel Consumption City (L/100km)',
                   'Fuel Consumption Hwy (L/100km)',
                   'Fuel Consumption Comb (L/100km)',
                   'Fuel Consumption Comb (mpg)',
                   'Engine Size (L)',
                   'Cylinders']
y_label = ['CO2 Emissions (g/km)']

ohe = OneHotEncoder()
fuel = np.array(ohe.fit_transform(input[[X_labels[6]]]).toarray())
y = input[y_label].to_numpy()
X = input[X_labels[0:6]].to_numpy()
X = np.append(X, fuel, axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

linearModel = lm.LinearRegression()
linearModel.fit(X_train, y_train)
print(linearModel.coef_)

y_test_p = linearModel.predict(X_test)
plt.scatter(x = y_test,
            y = y_test_p,
            s = 20,
            c = '#0000ff')
plt.plot(y_test, y_test, color = 'black', linestyle = 'dashed')
plt.xlabel("Test values")
plt.ylabel("Predicted values")

error = abs(y_test - y_test_p)
max_error = np.argmax(error)
print(error[max_error])
print(input.iloc[max_error][['Make', 'Model']])

plt.show()