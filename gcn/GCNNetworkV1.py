import numpy as np
from utils import load_data
import utils
seed = 2
class GCNNetwork:
    def __init__(self):
        self.n_x = None
        self.n_y = None
        self.n_h = None
        self.m = None
        self.W1 = None
        self.W2 = None
        self.X = None
        self.Y =None

    def layer_sizes(self, X, Y, hidden_dim) -> tuple:
        """
        Arguments:
        X -- input dataset of shape (input size, number of examples)
        Y -- labels of shape (output size, number of examples)

        Returns:
        n_x -- the size of the input layer
        n_h -- the size of the hidden layer
        n_y -- the size of the output layer
        """
        self.X = X
        self.Y = Y
        self.m = X.shape[0]  #代表样本个数
        self.n_x = X.shape[1] # size of input layer, 代表样本特征个数
        self.n_h = hidden_dim
        self.n_y = Y.shape[1] # size of output layer
        return self.n_x, self.n_h, self.n_y

    def initialize_parameters(self, A) -> dict:
        """
        Argument:
        n_x -- size of the input layer
        n_h -- size of the hidden layer
        n_y -- size of the output layer

        Returns:
        params -- python dictionary containing your parameters:
                       W1 -- weight matrix of shape (n_h, n_x)
                       W2 -- weight matrix of shape (n_y, n_h)
        """
        self.A = A
        np.random.seed(seed)
        #self.W1 = np.random.rand(self.n_x, self.n_h)
        self.W1 = np.random.uniform(-np.sqrt(1./self.n_h), np.sqrt(1./self.n_h), (self.n_x, self.n_h))
        #self.W2 = np.random.rand(self.n_h, self.n_y)
        self.W2 = np.random.uniform(-np.sqrt(1./self.n_h), np.sqrt(1./self.n_y), (self.n_h, self.n_y))
        assert (self.W1.shape == (self.n_x, self.n_h))
        assert (self.W2.shape == (self.n_h, self.n_y))
        parameters = {"W1": self.W1,
                      "W2": self.W2
                      }
        return parameters

    def foward_propagation(self, A, X, parameters) -> dict:
        W1 = parameters["W1"]
        W2 = parameters["W2"]
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

    def backward_propagation(self, parameters, cache, X, Y, train_mask) -> dict:

        W1 = parameters["W1"]
        W2 = parameters["W2"]


        OUT1 = cache["OUT1"]
        OUT2 = cache["OUT2"]

        OUT2[~train_mask] = Y[~train_mask]

        dL_dZ2 = OUT2 - Y
        dL_dW2 = np.dot(np.dot(self.A, OUT1).transpose(), dL_dZ2)
        temp = np.dot(np.dot(self.A.transpose(), dL_dZ2), W2.transpose())
        dL_dZ1 =  temp * utils.relu_diff(OUT1)
        dL_dW1 = np.dot(np.dot(self.A, X).transpose(), dL_dZ1)

        grads = {
            "dW1" : dL_dW1,
            "dW2" : dL_dW2
        }
        return grads


    def update_grads(self, parameters, grads, learning_rate = 1.2):
        W1 = parameters["W1"]
        W2 = parameters["W2"]

        dW1 = grads["dW1"]
        dW2 = grads["dW2"]

        self.W1 = W1 - learning_rate * dW1
        self.W2 = W2 - learning_rate * dW2

        parameters = {
            "W1" : self.W1,
            "W2" : self.W2
        }
        return parameters


    def calc_loss(self, X, Y, A, mask):
        N = mask.sum()
        preds = self.forward(X, A)
        loss = np.sum(Y[mask] * np.log(preds[mask]))
        loss = np.asscalar(-loss) / N

        return loss
    def compute_cost(self, OUT2, Y, TRAIN_MASK) -> float:
        '''
        计算损失函数
        :param OUT2:
        :param Y:
        :return:
        cost float
        '''
        m = TRAIN_MASK.sum()
        print("m = " + str(m))
        loss = np.sum(Y[TRAIN_MASK] * np.log(OUT2[TRAIN_MASK]))
        loss = np.asscalar(-loss) / m
        # logprobs = np.log(OUT2[TRAIN_MASK])
        # cost = np.sum(-1 * np.dot(Y[TRAIN_MASK], logprobs.T))
        # cost = np.squeeze(cost)     # makes sure cost is the dimension we expect.
        # # E.g., turns [[17]] into 17
        # assert(isinstance(cost, float))
        return loss
    def loss_accuracy(self, OUT2, Y, TARIN_MASK):
        """ Combination of calc_loss and compute_accuracy to reduce the need to forward propagate twice """
        # from calc_loss
        N = TARIN_MASK.sum()

        loss = np.sum(Y[TARIN_MASK] * np.log(OUT2[TARIN_MASK]))
        loss = np.asscalar(-loss) / N
        loss = np.sum(Y[TARIN_MASK] * np.log(OUT2[TARIN_MASK]))
        loss = np.asscalar(-loss) / N

        # from compute_accuracy
        out = OUT2
        out_class = np.argmax(out[TARIN_MASK], axis=1)
        expected_class = np.argmax(Y[TARIN_MASK], axis=1)
        num_correct = np.sum(out_class == expected_class).astype(float)
        accuracy = num_correct / expected_class.shape[0]

        return loss, accuracy
    def loss_accuracy_2(self, X, Y, A, mask, parameters):
        """ Combination of calc_loss and compute_accuracy to reduce the need to forward propagate twice """

        # from calc_loss
        N = mask.sum()
        preds = self.foward_propagation(A,X,parameters)["OUT2"]
        loss = np.sum(Y[mask] * np.log(preds[mask]))
        loss = np.asscalar(-loss) / N



        # from compute_accuracy
        out = preds
        out_class = np.argmax(out[mask], axis=1)
        expected_class = np.argmax(Y[mask], axis=1)
        num_correct = np.sum(out_class == expected_class).astype(float)
        accuracy = num_correct / expected_class.shape[0]

        return loss, accuracy
if __name__ == '__main__' :
    A = np.random.randn(5,5)
    X_assess = np.random.randn(5, 2)
    Y_assess = np.random.randn(5, 1)
    gcn = GCNNetwork()
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask,yy_train = load_data("cora")
    gcn.layer_sizes(features, yy_train, 8)
    print("The size of 样本 is: n_x = " + str(gcn.m))
    print("The size of the input layer is: n_x = " + str(gcn.n_x))
    print("The size of the hidden layer is: n_x = " + str(gcn.n_h))
    print("The size of the output layer is: n_x = " + str(gcn.n_y))
    parameters = gcn.initialize_parameters(adj)
    print(parameters.get("W1").shape)
    print(parameters.get("W2").shape)


    for i in range(100):
        cache = gcn.foward_propagation(gcn.A, gcn.X, parameters)
        #print("cache 前向传播结果： "  + str(cache))

        # cost =  gcn.compute_cost(cache['OUT2'], y_train, train_mask)
        # print("cost = %f" % cost)
        train_loss, train_accuracy = gcn.loss_accuracy(cache['OUT2'], y_train,train_mask)
        val_loss, val_accuracy = gcn.loss_accuracy_2(gcn.X, y_val,  gcn.A, val_mask, parameters)

        print("Epoch:", '%04d' % (i + 1), "train_loss=", "{:.5f}".format(train_loss),
              "train_acc=", "{:.5f}".format(train_accuracy), "val_loss=", "{:.5f}".format(val_loss),
              "val_acc=", "{:.5f}".format(val_accuracy))
        grads = gcn.backward_propagation(parameters, cache, features, y_train, train_mask)

        parameters = gcn.update_grads(parameters, grads, 0.14750295365340862)
        #print(parameters)


