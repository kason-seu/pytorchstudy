from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
iris = datasets.load_iris()

print(iris.keys())

#print(iris.DESCR)


print(iris.data.shape)
print(iris.data.shape[1])
print(iris.target.shape)


x = iris.data[:,:2]
plt.scatter(x[:,0], x[:,1])
plt.show()


y = iris.target

for i in range(iris.data.shape[1]):
    print(i)
    if (i == 0) :
        plt.scatter(x[y==i, 0], x[y==i, 1], marker='o')
    elif(i == 1) :
        plt.scatter(x[y==i, 0], x[y==i, 1], marker='+')
    elif(i == 2):
        plt.scatter(x[y==i, 0], x[y==i, 1], marker='*')
    else:
        plt.scatter(x[y==i, 0], x[y==i, 1], marker='^')

plt.show()

x = iris.data[:,2:]
for i in range(iris.data.shape[1]):
    print(i)
    if (i == 0) :
        plt.scatter(x[y==i, 0], x[y==i, 1], marker='o')
    elif(i == 1) :
        plt.scatter(x[y==i, 0], x[y==i, 1], marker='+')
    elif(i == 2):
        plt.scatter(x[y==i, 0], x[y==i, 1], marker='*')
    else:
        plt.scatter(x[y==i, 0], x[y==i, 1], marker='^')

plt.show()