import pandas as pd
import matplotlib.pyplot as plt
import math

data = pd.read_csv('data_C02_emission.csv')

#a)
#Pomocu histograma prikažite emisiju C02 plinova.
data['CO2 Emissions (g/km)'].plot(kind = "hist")
plt.xlabel('CO2 emissions (g/km)')
plt.title('CO2 emissions histogram')
plt.xlim([0, math.floor(data['CO2 Emissions (g/km)'].max()/100)*100])
plt.grid(True)

#b)
#Pomocu dijagrama raspršenja prikažite odnos izmedu gradske potrošnje goriva i emisije C02 plinova.
scatter = data.copy()
for i in range(len(scatter['Fuel Type'])):
    scatter['Fuel Type'][i] = ord(scatter['Fuel Type'][i])/ord('Z')
scatter.plot.scatter(
    x = 'Fuel Consumption City (L/100km)',
    y = 'CO2 Emissions (g/km)',
    c = 'Fuel Type',
    cmap = 'Set1',
    s = 10)

#c)
#Pomocu kutijastog dijagrama prikažite razdiobu izvangradske potrošnje s obzirom na tip goriva.
data.boxplot(
    column = ['Fuel Consumption Hwy (L/100km)'],
    by = 'Fuel Type')

#d)
#Pomocu stupcastog dijagrama prikažite broj vozila po tipu goriva. 
plt.figure()
plt.subplot(211)
data.groupby('Fuel Type').size().plot(kind = 'bar')

#e)
#Pomocu stupcastog grafa prikažite na istoj slici prosjecnu C02 emisiju vozila s obzirom na broj cilindara.
plt.subplot(212)
data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean().plot(kind = 'bar')
plt.show()



