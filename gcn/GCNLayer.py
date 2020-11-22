import numpy as np

from utils import load_data
class GCNLayer:

    def __init__(self, A, X, Y, hidden_dim):
        self.A = A
        self.X = X
        self.Y = Y
        self.layer_sizes(X,Y, hidden_dim)
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
        self.n_x = X.shape[0] # size of input layer, 代表样本数目
        self.n_h = hidden_dim
        self.n_y = Y.shape[0] # size of output layer
        return (self.n_x, self.n_h, self.n_y)
    def __str__(self):
        return "n_x=%d, n_h=%d, n_y=%d" % (self.n_x, self.n_h,self.n_y)

if __name__ == '__main__':
    np.random.seed(1)
    A = np.random.randn(5,5)
    X_assess = np.random.randn(5, 3)
    Y_assess = np.random.randn(2, 3)
    gcnlayer = GCNLayer(A,X_assess,Y_assess,4)
    print("The size of the input layer is: n_x = " + str(gcnlayer.n_x))
    print("The size of the hidden layer is: n_x = " + str(gcnlayer.n_h))
    print("The size of the output layer is: n_x = " + str(gcnlayer.n_y))


    print(gcnlayer)

