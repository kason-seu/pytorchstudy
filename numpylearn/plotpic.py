
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

x = np.linspace(1,10,100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1,color='red',linestyle = '--', label = 'sinx(x)')
plt.plot(x, y2, color = 'green', label = 'cos(x)')
plt.axis([5,15, -1,1])
plt.xlabel('x---label')
plt.ylabel('y---label')
plt.legend()
plt.title('welcome to matplot, sinx(x) with cos(x)')
plt.show()


plt.scatter(x, y1,color = 'blue')
plt.scatter(x, y2,color = 'red')
plt.show()

x = np.random.normal(0,100, 1000)
y = np.random.normal(0,100,1000)

plt.scatter(x,y, alpha=0.4)
plt.show()