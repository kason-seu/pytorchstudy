from GCNNetworkProd import GCNNetwork
import time
import utils
def gcnTest():
    # self.m = X.shape[0]  #代表样本个数
    # self.n_x = X.shape[1] # size of input layer, 代表样本特征个数
    # self.n_h = hidden_dim
    # self.n_y = Y.shape[1] # size of output layer

    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask,yy_train = utils.load_data("cora")
    m = features.shape[0]
    n_x = features.shape[1]
    n_h = 8
    n_y = y_train.shape[1]
    A = adj
    X = features

    gcn = GCNNetwork(m, n_x, n_h, n_y)
    print("The size of 样本 is: n_x = " + str(gcn.m))
    print("The size of the input layer is: n_x = " + str(gcn.n_x))
    print("The size of the hidden layer is: n_x = " + str(gcn.n_h))
    print("The size of the output layer is: n_x = " + str(gcn.n_y))

    print(gcn.parameters.get("W1").shape)
    print(gcn.parameters.get("W2").shape)

    epochs = 500
    for i in range(epochs):
        start = time.time()
        cache = gcn.foward_propagation(A, X)
        #print("cache 前向传播结果： %s" % str(cache))
        train_loss, train_accuracy = gcn.loss_accuracy(cache['OUT2'], y_train,train_mask)
        val_loss, val_accuracy = gcn.loss_accuracy_test(A, X, y_val, val_mask)
        grads = gcn.backward_propagation(A, cache, features, y_train, train_mask)
        parameters = gcn.update_grads(grads, 0.14750295365340862)
        end = time.time()
        print("Epoch:", '%04d' % (i + 1), "train_loss=", "{:.5f}".format(train_loss),
              "train_acc=", "{:.5f}".format(train_accuracy), "val_loss=", "{:.5f}".format(val_loss),
              "val_acc=", "{:.5f}".format(val_accuracy), " cost time = %d" % (end - start))
        #print("changed grads： %s" % str(grads), "changed w： %s" % str(parameters))

if __name__ == '__main__' :
    gcnTest()



