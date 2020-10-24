
import numpy as np

import matplotlib.pyplot as plt
from math import sqrt
from collections import Counter
def knn_classfier(k, x_train, y_train, x):
    assert 1 <= k <= x_train.shape[0], "k 必须在1和总数之间"
    distance = [ sqrt(np.sum((xx-x) ** 2)) for xx in x_train]
    distinceIndex = np.argsort(distance)
    votesTopK = [y[i] for i in distinceIndex[:k]]
    votes = Counter(votesTopK)
    return votes.most_common(1)[0][0]
raw_data_X = [[3.393533211, 2.331273381],
              [3.110073483, 1.781539638],
              [1.343808831, 3.368360954],
              [3.582294042, 4.679179110],
              [2.280362439, 2.866990263],
              [7.423436942, 4.696522875],
              [5.745051997, 3.533989803],
              [9.172168622, 2.511101045],
              [7.792783481, 3.424088941],
              [7.939820817, 0.791637231]
              ]
raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

x1 = np.array(raw_data_X)
y = np.array(raw_data_y)

plt.scatter(x1[y==0, 0], x1[y==0, 1], color='black')
plt.scatter(x1[y==1,0],x1[y==1, 1],color='green')


x = np.array([8.093607318, 3.365731514])

plt.scatter(x[0],x[1], color='red', marker='+')
plt.show()

predict_y = knn_classfier(6, x1, y, x)
print(predict_y)