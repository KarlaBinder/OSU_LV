import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn . linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

labels= {0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    edgecolor = 'w',
                    label=labels[cl])

# ucitaj podatke
df = pd.read_csv("penguins.csv")

# izostale vrijednosti po stupcima
print(df.isnull().sum())

# spol ima 11 izostalih vrijednosti; izbacit cemo ovaj stupac
df = df.drop(columns=['sex'])

# obrisi redove s izostalim vrijednostima
df.dropna(axis=0, inplace=True)

# kategoricka varijabla vrsta - kodiranje
df['species'].replace({'Adelie' : 0,
                        'Chinstrap' : 1,
                        'Gentoo': 2}, inplace = True)

print(df.info())

# izlazna velicina: species
output_variable = ['species']

# ulazne velicine: bill length, flipper_length
input_variables = ['bill_length_mm',
                    'flipper_length_mm']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()

# podjela train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

#a) Pomocu stupcastog dijagrama prikažite koliko primjera postoji za svaku klasu (vrstu pingvina) u skupu podataka za ucenje i skupu podataka za testiranje. Koristite numpy funkciju unique.

unique_train, counts_train = np.unique(y_train, return_counts=True)
unique_test, counts_test = np.unique(y_test, return_counts=True)

fig,ax = plt.subplots()
ax.bar(unique_train, counts_train, align='center', alpha=0.5, label='Train')
ax.bar(unique_test, counts_test, align='center', alpha=0.5, label='Test')
ax.set_xticks(unique_train)
ax.set_xticklabels([labels[c] for c in unique_train])
ax.set_xlabel('Species')
ax.set_ylabel('Number of examples')
ax.legend()
plt.show()

#b) Izgradite model logisticke regresije pomocu scikit-learn biblioteke na temelju skupa podataka za ucenje.
LogRegression_model = LogisticRegression(random_state=123, max_iter=1000)
LogRegression_model.fit(X_train, y_train.ravel())

#c) Pronadite u atributima izgradenog modela parametre modela. Koja je razlika u odnosu na binarni klasifikacijski problem iz prvog zadatka?
print('Koeficijenti:', LogRegression_model.coef_)
print('Intercept:', LogRegression_model.intercept_)

#d) Pozovite funkciju plot_decision_region pri cemu joj predajte podatke za ucenje i izgradeni model logisticke regresije. Kako komentirate dobivene rezultate?
plot_decision_regions(X_train, y_train.ravel(), LogRegression_model)
plt.legend()
plt.show()

#e) Provedite klasifikaciju skupa podataka za testiranje pomocu izgradenog modela logisticke regresije. Izracunajte i prikažite matricu zabune na testnim podacima. Izracunajte tocnost.
#Pomocu classification_report funkcije izracunajte vrijednost cetiri glavne metrike

y_test_p=LogRegression_model.predict(X_test)
cm = confusion_matrix(y_test, y_test_p)
class_report = classification_report(y_test, y_test_p)

print('Confusion Matrix:',cm)
print('\nClassification Report:',class_report)

