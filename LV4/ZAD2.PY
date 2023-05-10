import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import sklearn.linear_model as lm
import sklearn.metrics as skmetrics

data = pd.read_csv("data_C02_emission.csv")
#data =data.drop(["Make", 'Vehicle Class', 'Transmission'], axis=1)
X_labels = ['Engine Size (L)',
            'Fuel Consumption City (L/100km)',
           'Cylinders',
           'Fuel Consumption Hwy (L/100km)',
           'Fuel Consumption Comb (L/100km)',
           'Fuel Consumption Comb (mpg)',
           'Fuel Type']
y_label = ['CO2 Emissions (g/km)']

ohe = OneHotEncoder()
#fit_transform ocekuje 2d array(dataframe[[stupac(i)]]), ne moze 1d (series[stupac])
X_enc = ohe.fit_transform(data[['Fuel Type']]).toarray()
data['Fuel Type'] = X_enc
X=data[X_labels].to_numpy()
y=data[y_labels].to_numpy()
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
plt.show()

abs = abs(y_test - y_test_p)
max = np.argmax(abs)
print(abs[max])
print(data.at[data.index[max], 'Model'])

