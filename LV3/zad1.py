import pandas as pd

data = pd.read_csv('data_C02_emission.csv')

#a)
#Koliko mjerenja sadrži DataFrame?
print('Broj mjerenja: ', len(data))

#Kojeg je tipa svaka velicina?
print(data.info())

#Postoje li izostale ili duplicirane vrijednosti?
dropna = data.isnull().sum()
print('Broj izostalih vrijednosti:')
print(dropna)
if sum(dropna) != 0:
    data.dropna(axis = 0)
    data.dropna(axis = 1)

data.drop_duplicates()
data = data.reset_index(drop = True)
print('Broj mjerenja nakon ciscenja: ', len(data))

#Kategoricke velicine konvertirajte u tip category
data[data.select_dtypes(['object']).columns]=data.select_dtypes(['object']).apply(lambda x:x.astype('category'))
print(data.info())

#b)
#Koja tri automobila ima najvecu odnosno najmanju gradsku potrošnju?
data = data.sort_values(by = ['Fuel Consumption City (L/100km)'])
print('Tri najmanje gradske potrosnje:')
print(data.head(3)[['Make', 'Model', 'Fuel Consumption City (L/100km)']])
print('Tri najvece gradske potrosnje:')
print(data.tail(3)[['Make', 'Model', 'Fuel Consumption City (L/100km)']])

#c)
#Koliko vozila ima velicinu motora izmedu 2.5 i 3.5 L?
byengine = data[(data['Engine Size (L)'] >= 2.5) & (data['Engine Size (L)'] <= 3.5)]
print('Broj vozila s motorom velicine 2.5<=m<=3.5: ', len(byengine))

#Kolika je prosjecna C02 emisija plinova za ova vozila?
print(byengine[['CO2 Emissions (g/km)']].mean())

#d)
#Koliko mjerenja se odnosi na vozila proizvodaca Audi?
audi = data[data['Make'] == 'Audi']
print('Broj Audija: ', len(audi))

#Kolika je prosjecna emisija C02 plinova automobila proizvodaca Audi koji imaju 4 cilindara?
audi4c = audi[audi['Cylinders'] == 4]
print('Broj Audija s 4 cilindra: ', len(audi4c))
print(audi4c[['CO2 Emissions (g/km)']].mean())

#e)
#Koliko je vozila s 4,6,8. . . cilindara?
four_cylinder = data[data['Cylinders']==4]
print(f'Vozilo s 4 cilindra:{four_cylinder}')

six_cylinder = data[data['Cylinders']==6]
print(f'Vozilo s 4 cilindra:{six_cylinder}')

eight_cylinder = data[data['Cylinders']==8]
print(f'Vozilo s 4 cilindra:{eight_cylinder}')

ten_cylinder = data[data['Cylinders']==10]
print(f'Vozilo s 4 cilindra:{ten_cylinder}')

twelve_cylinder = data[data['Cylinders']==12]
print(f'Vozilo s 4 cilindra:{twelve_cylinder}')

sixteen_cylinder = data[data['Cylinders']==16]
print(f'Vozilo s 4 cilindra:{sixteen_cylinder}')

#Kolika je prosjecna emisija C02 plinova s obzirom na broj cilindara?
bycylinder = data.groupby('Cylinders')
print(bycylinder[['CO2 Emissions (g/km)']].mean())

#f)
#Kolika je prosjecna gradska potrošnja u slucaju vozila koja koriste dizel, a kolika za vozila koja koriste regularni benzin?
diesel = data[data['Fuel Type'] == 'D']
print('Prosjecna gradska potrosnja na dizel: ', diesel['Fuel Consumption City (L/100km)'].mean())

#Koliko iznose medijalne vrijednosti?
reg_gas = data[data['Fuel Type'] == 'X']
print('Prosjecna gradska potrosnja na benzin: ', reg_gas['Fuel Consumption City (L/100km)'].mean())
print('Medijalna gradska potrosnja na dizel: ', diesel['Fuel Consumption City (L/100km)'].median())
print('Medijalna gradska potrosnja na benzin: ', reg_gas['Fuel Consumption City (L/100km)'].median())

#g)
#Koje vozilo s 4 cilindra koje koristi dizelski motor ima najvecu gradsku potrošnju goriva?
maxconsumer = data[(data['Cylinders'] == 4) & (data['Fuel Type'] == 'D')]
maxconsumer = maxconsumer[maxconsumer['Fuel Consumption City (L/100km)'] == maxconsumer['Fuel Consumption City (L/100km)'].max()]
print(maxconsumer[['Make', 'Model']])

#h)
#Koliko ima vozila ima rucni tip mjenjaca (bez obzira na broj brzina)?
manual = data[data['Transmission'].str.startswith('M')]
print('Broj rucnih mjenjaca: ', len(manual))

#i)
#Izracunajte korelaciju izmedu numerickih velicina.
pd.set_option('display.max_columns', None)
print(data.corr(numeric_only = True))