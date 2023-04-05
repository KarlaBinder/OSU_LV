import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn . linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

#a) Prikažite podatke za ucenje u x1 − x2 ravnini matplotlib biblioteke pri cemu podatke obojite
#s obzirom na klasu. Prikažite i podatke iz skupa za testiranje, ali za njih koristite drugi
#marker (npr. ’x’). Koristite funkciju scatter koja osim podataka prima i parametre c i
#cmap kojima je moguce definirati boju svake klase.

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired,label='Training Data')
plt.scatter(X_test[:, 0], X_test[:, 1], marker='x', c=y_test, cmap=plt.cm.Paired,label='Testing Data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Binary Classification Problem')
plt.legend()
plt.show()

#b)  Izgradite model logisticke regresije pomocu scikit-learn biblioteke na temelju skupa poda-taka za ucenje
LogRegression_model = LogisticRegression()
LogRegression_model.fit(X_train, y_train)

#c) Pronadite u atributima izgradenog modela parametre modela. Prikažite granicu odluke
#naucenog modela u ravnini x1 − x2 zajedno s podacima za ucenje. Napomena: granica
#odluke u ravnini x1 − x2 definirana je kao krivulja: θ0 + θ1x1 + θ2x2 = 0.
theta0 = LogRegression_model.intercept_
theta1, theta2 = LogRegression_model.coef_[0]
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired)
x1 = np.linspace(X_train[:, 0].min()-1, X_train[:, 0].max()+1, 100)
x2 = (-1/theta2) * (theta0 + theta1*x1)
plt.plot(x1, x2, color='k', linestyle='--', alpha=0.4)
plt.xlim(X_train[:, 0].min()-1, X_train[:, 0].max()+1)
plt.ylim(X_train[:, 1].min()-1, X_train[:, 1].max()+1)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

#d) Provedite klasifikaciju skupa podataka za testiranje pomocu izgradenog modela logisticke
#regresije. Izracunajte i prikažite matricu zabune na testnim podacima. Izracunate tocnost,
#preciznost i odziv na skupu podataka za testiranje.

y_test_p = LogRegression_model.predict(X_test)
cm = confusion_matrix(y_test, y_test_p)
print("Matrica zabune:\n", cm)
disp = ConfusionMatrixDisplay(confusion_matrix(y_test,y_test_p))
disp.plot()
plt.show()
print(classification_report(y_test,y_test_p))

#e) Prikažite skup za testiranje u ravnini x1 − x 2 . Zelenom bojom oznacite dobro klasificirane
#primjere dok pogrešno klasificirane primjere oznacite crnom bojom.

plt.scatter(X_test[y_test_p==y_test, 0], X_test[y_test_p==y_test, 1], c='green', alpha=0.5)
plt.scatter(X_test[y_test_p!=y_test, 0], X_test[y_test_p!=y_test, 1], c='black', alpha=0.5)
plt.xlim(X_test[:, 0].min()-1, X_test[:, 0].max()+1)
plt.ylim(X_test[:, 1].min()-1, X_test[:, 1].max()+1)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()