import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as lm
import sklearn.metrics as skmetrics

#a) Odaberite željene numericke velicine specificiranjem liste s nazivima stupaca. Podijelite
#podatke na skup za ucenje i skup za testiranje u omjeru 80%-20%.
data = pd.read_csv("data_C02_emission.csv")
input_variables = ['Fuel Consumption City (L/100km)',
                   'Fuel Consumption Hwy (L/100km)',
                   'Fuel Consumption Comb (L/100km)',
                   'Fuel Consumption Comb (mpg)',
                   'Engine Size (L)',
                   'Cylinders']

output_variable = ['CO2 Emissions (g/km)']
X = data[input_variables].to_numpy()
y = data[output_variable].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

#b) Pomocu matplotlib biblioteke i dijagrama raspršenja prikažite ovisnost emisije C02 plinova
#o jednoj numerickoj velicini. Pri tome podatke koji pripadaju skupu za ucenje oznacite
#plavom bojom, a podatke koji pripadaju skupu za testiranje oznacite crvenom bojom.
X_train_tsposed = np.transpose(X_train)
X_test_tsposed = np.transpose(X_test)
for i in range(len(input_variables)):
    plt.scatter(x = X_train_tsposed[i],
                y = y_train,
                s = 20,
                c = 'b',
                label = 'Training')
    plt.scatter(x = X_test_tsposed[i],
                y = y_test,
                s = 20,
                c = 'r',
                label = 'Test')
    plt.xlabel(input_variables[i])
    plt.ylabel(output_variable[0])
    plt.legend()
    plt.figure()

#c)Izvršite standardizaciju ulaznih velicina skupa za ucenje. Prikažite histogram vrijednosti
#jedne ulazne velicine prije i nakon skaliranja. Na temelju dobivenih parametara skaliranja
#transformirajte ulazne velicine skupa podataka za testiranje
ss = StandardScaler()
X_train_s = ss.fit_transform(X_train)
X_train_s_tsposed = np.transpose(X_train_s)
for i in range(len(input_variables)):
    plt.subplot(211)
    plt.hist(X_train_tsposed[i])
    plt.title(input_variables[i] + " before fit")
    plt.subplot(212)
    plt.hist(X_train_s_tsposed[i])
    plt.title(input_variables[i] + " after fit")
    plt.figure()
X_test_s = ss.transform(X_test)

#d) Izgradite linearni regresijski modeli. Ispišite u terminal dobivene parametre modela i povežite ih s izrazom 4.6.
linearModel = lm.LinearRegression()
linearModel.fit(X_train_s, y_train)
print(linearModel.coef_)

#e) Izvršite procjenu izlazne veliˇcine na temelju ulaznih veliˇcina skupa za testiranje. Prikažite
#pomocu dijagrama raspršenja odnos izme du stvarnih vrijednosti izlazne velicine i procjene dobivene modelom.
y_test_p = linearModel.predict(X_test_s)
plt.scatter(x = y_test,
            y = y_test_p,
            s = 20,
            c = '#b')
plt.plot(y_test, y_test, color = 'black', linestyle = 'dashed')
plt.xlabel("Test values")
plt.ylabel("Predicted values")

#f) Izvršite vrednovanje modela na nacin da izracunate vrijednosti regresijskih metrika na skupu podataka za testiranje.
MSE = skmetrics.mean_squared_error(y_test, y_test_p)
print("Test size: 0.2")
print("MSE: ", MSE)
print("RMSE: ", math.sqrt(MSE))
print("MAE: ", skmetrics.mean_absolute_error(y_test, y_test_p))
print("MAPE: ", skmetrics.mean_absolute_percentage_error(y_test, y_test_p))
print("Rsquared: ", skmetrics.r2_score(y_test, y_test_p))

#g) Što se dogada s vrijednostima evaluacijskih metrika na testnom skupu kada mijenjate broj ulaznih velicina?
sample = 0.2
for i in range(10):
    sample += 0.05
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = sample, random_state = 1)
    ss = StandardScaler()
    X_train_s = ss.fit_transform(X_train)
    X_test_s = ss.transform(X_test)
    linearModel = lm.LinearRegression()
    linearModel.fit(X_train_s, y_train)
    y_test_p = linearModel.predict(X_test_s)
    MSE = skmetrics.mean_squared_error(y_test, y_test_p)
    print("Test size: ", sample)
    print("MSE: ", MSE)
    print("RMSE: ", math.sqrt(MSE))
    print("MAE: ", skmetrics.mean_absolute_error(y_test, y_test_p))
    print("MAPE: ", skmetrics.mean_absolute_percentage_error(y_test, y_test_p))
    print("Rsquared: ", skmetrics.r2_score(y_test, y_test_p))

plt.show()