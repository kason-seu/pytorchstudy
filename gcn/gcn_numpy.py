import numpy as np

# 创建图的邻接矩阵
A = np.array([[0,1,0,0],[0,0,1,1],[0,1,0,0],[1,0,1,0]], dtype=np.float)
I = np.eye(A.shape[0])

A_hat = A + I

# 为图里面的每个节点根据其索引生成两个整数特征。
X = np.array([[i,-i] for i in range(A.shape[0])], dtype=np.float)
print(X)

# 传播
print(np.dot(A_hat,X))


W = np.array([[1,-1],[-1,1]])

D_hat = np.diag(np.sum(A_hat, axis = 0))
print(np.dot(np.dot(np.dot(np.linalg.inv(D_hat), A_hat), X),W))
F = np.dot(np.dot(np.dot(np.linalg.inv(D_hat), A_hat), X),W)
print(1 * (F > 0) * F)





import networkx as nx
from networkx import to_numpy_matrix
zkc = nx.karate_club_graph()
order = sorted(list(zkc.nodes()))
A = to_numpy_matrix(zkc, nodelist=order)
I = np.eye(zkc.number_of_nodes())
A_hat = A + I
D_hat = np.array(np.sum(A_hat, axis=0))[0]
D_hat = np.matrix(np.diag(D_hat))