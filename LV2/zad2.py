import numpy as np
import matplotlib.pyplot as plt

arra= np.loadtxt('data.csv', delimiter=',', skiprows=1, dtype=float)
#imamo 2 načina importanja datoteka preko numpyja loadtxt i genfromtxt
#arra=np.genfromtxt('data.csv', delimiter=',',dtype= None, encoding=None,skip_header=1)

#a) Na koliko je osoba izvršeno mjerenja?
size=len(arra)
print('Izvrseno je mjenjenja na:', size)
size=int(size)           #ili arra.shape

#b) Prikaži odnos visine i mase
x=np.array(arra[:,1])      #da smo koristili genfromtxt ovako bi izgledalo x=arra[:,1] 
y=np.array(arra[:,2])
plt.title('Odnos svih visina i tezina')
plt.scatter(x,y,alpha=0.5,c='b', linewidths=1)
plt.show()

#c) Isto kao b) ali za svaku 50-tu osobu
x1=x[::50]    #f=df[::50,1]
y1=y[::50]    #g=df[::50,2]
plt.title('Odnos visine i tezine svake 50 osobe')
plt.scatter(x1,y1,alpha=0.5,c='y', linewidths=1)
plt.show()

#c) U terminalu ispiši min, max i srednju vrijednost visine
#height=df[:,1]
#print(f'Minimalna visina {np.min(height)} Maksimalna visina {np.max(height)} Srednja vrijednost {np.mean(height)}')
print('Min', x.min())
print('Max:', x.max())
print('Srednja vrijednost:', x.mean())

#d) isto kao i c) samo razdvojiti muškarce i žene
male=(arra[:,0]==1)
female=(arra[:,0]==0)

print('Min muski:', arra[male,1].min())
print('Max muski:', arra[male,1].max())
print('Srednja vrijednost muski:', arra[male,1].mean())

print('Min zene:', arra[female,1].min())
print('Max zene:', arra[female,1].max())
print('Srednja vrijednost zene:', arra[female,1].mean())
