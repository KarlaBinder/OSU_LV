import numpy as np
import matplotlib.pyplot as plt
x=np.array([1,2,3,3,1])
y=np.array([1,2,2,1,1])
plt.plot(x,y,'y',linewidth=2,marker=".",markersize=6)
plt.axis([0,4,0,4])
plt.xlabel('x os')
plt.ylabel('y os')
plt.show()



