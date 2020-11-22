import numpy as np
import utils
class GCNNetwork:
    def __init__(self, m, n_x, n_h,n_y, seed = 2):
        '''
        初始化GCN网络
        :param m: 样本个数
        :param n_x: 样本特征个数，一层网络的size
        :param n_h: 隐藏层网络的size
        :param n_y: 输出层的size
        '''
        self.m = m
        self.n_x = n_x
        self.n_h = n_h
        self.n_y = n_y
        np.random.seed(seed)
        #self.W1 = np.random.rand(self.n_x, self.n_h)
        self.W1 = np.random.uniform(-np.sqrt(1./self.n_h), np.sqrt(1./self.n_h), (self.n_x, self.n_h))
        #self.W2 = np.random.rand(self.n_h, self.n_y)
        self.W2 = np.random.uniform(-np.sqrt(1./self.n_h), np.sqrt(1./self.n_y), (self.n_h, self.n_y))
        assert (self.W1.shape == (self.n_x, self.n_h))
        assert (self.W2.shape == (self.n_h, self.n_y))
        self.parameters = {"W1": self.W1,
                           "W2": self.W2
                          }

    def foward_propagation(self, A, X) -> dict:
        '''
        前向传播
        :param A: 邻接矩阵
        :param X: 特征矩阵
        :return:
        Z1: 隐藏层结果
        OUT1: 隐藏层结果经过非线性激活函数后的结果
        Z2：输出层结果
        OUT2: 输出层结果经过非线性激活函数之后的结果

        '''
        W1 = self.parameters["W1"]
        W2 = self.parameters["W2"]
        assert(A.shape[0] == self.m)
        assert(X.shape[0] == self.m)
        assert(W1.shape[0] == X.shape[1])
        Z1 = np.dot(np.dot(A, X), W1)
        OUT1 = utils.relu(Z1)

        assert(A.shape[0] == OUT1.shape[0])
        assert(OUT1.shape[1] == W2.shape[0])
        Z2 = np.dot(np.dot(A, OUT1), W2)
        OUT2 = utils.sigmod(Z2)
        cache = {
            "Z1":Z1,
            "OUT1":OUT1,
            "Z2":Z2,
            "OUT2":OUT2
        }
        return cache

    def backward_propagation(self, A, cache, X, Y, train_mask) -> dict:
        '''

        :param A: 链接矩阵
        :param cache: 正向传播的结果集
        :param X: 属性特征
        :param Y: 目标结果
        :param train_mask: 训练集索引
        :return:
        '''
        W1 = self.parameters["W1"]
        W2 = self.parameters["W2"]


        OUT1 = cache["OUT1"]
        OUT2 = cache["OUT2"]

        OUT2[~train_mask] = Y[~train_mask]

        dL_dZ2 = OUT2 - Y
        dL_dW2 = np.dot(np.dot(A, OUT1).transpose(), dL_dZ2)
        temp = np.dot(np.dot(A.transpose(), dL_dZ2), W2.transpose())
        dL_dZ1 =  temp * utils.relu_diff(OUT1)
        dL_dW1 = np.dot(np.dot(A, X).transpose(), dL_dZ1)

        grads = {
            "dW1" : dL_dW1,
            "dW2" : dL_dW2
        }
        return grads


    def update_grads(self, grads, learning_rate = 1.2):
        '''
        梯度更新
        :param grads:
        :param learning_rate: 学习率
        :return:
        '''
        W1 = self.parameters["W1"]
        W2 = self.parameters["W2"]

        dW1 = grads["dW1"]
        dW2 = grads["dW2"]

        self.W1 = W1 - learning_rate * dW1
        self.W2 = W2 - learning_rate * dW2

        self.parameters = {
            "W1" : self.W1,
            "W2" : self.W2
        }
        return self.parameters

    def compute_cost(self, OUT2, Y, train_mask) -> float:
        '''
        计算损失函数
        :param OUT2:
        :param Y:
        :return:
        cost float
        '''
        m = train_mask.sum()
        logprobs = np.log(OUT2[train_mask])
        cost = np.sum(-1 * Y[train_mask] * logprobs) / m
        cost = np.squeeze(cost)     # makes sure cost is the dimension we expect.
        # # E.g., turns [[17]] into 17
        assert(isinstance(cost, float))
        return cost
    def loss_accuracy(self, OUT2, Y, train_mask):
        # from compute_cost
        loss = self.compute_cost(OUT2, Y, train_mask)
        # compute_accuracy
        out = OUT2
        out_class = np.argmax(out[train_mask], axis=1)
        expected_class = np.argmax(Y[train_mask], axis=1)
        num_correct = np.sum(out_class == expected_class).astype(float)
        accuracy = num_correct / expected_class.shape[0]
        return loss, accuracy

    def loss_accuracy_test(self, A, X, Y, mask):
        '''测试集的验证'''
        OUT2 = self.foward_propagation(A,X)["OUT2"]
        loss = self.compute_cost(OUT2, Y, mask)
        # compute_accuracy
        out = OUT2
        out_class = np.argmax(out[mask], axis=1)
        expected_class = np.argmax(Y[mask], axis=1)
        num_correct = np.sum(out_class == expected_class).astype(float)
        accuracy = num_correct / expected_class.shape[0]

        return loss, accuracy