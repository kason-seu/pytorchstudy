
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import torch


w  = torch.tensor(1.5, requires_grad=True)

def forward(x):
    return w * x

def criterion(YHAT,Y) :
    return torch.mean((YHAT-Y) ** 2)

def backward(loss):
    loss.backward()


# The class for plotting

class plot_diagram():

    # Constructor
    def __init__(self, X, Y, w, stop, go = False):
        start = w.data
        self.error = []
        self.parameter = []
        self.X = X.numpy()
        self.Y = Y.numpy()
        self.parameter_values = torch.arange(start, stop)
        self.Loss_function = [criterion(forward(X), Y) for w.data in self.parameter_values]
        w.data = start

    # Executor
    def __call__(self, Yhat, w, error, n):
        self.error.append(error)
        self.parameter.append(w.data)
        plt.subplot(212)
        plt.plot(self.X, Yhat.detach().numpy())
        plt.plot(self.X, self.Y,'ro')
        plt.xlabel("A")
        plt.ylim(-20, 20)
        plt.subplot(211)
        plt.title("Data Space (top) Estimated Line (bottom) Iteration " + str(n))
        plt.plot(self.parameter_values.numpy(), self.Loss_function)
        plt.plot(self.parameter, self.error, 'ro')
        plt.xlabel("B")
        plt.figure()

    # Destructor
    def __del__(self):
        plt.close('all')



X = torch.range(-3,3,0.1).view(-1,1)

F = 3 * X

plt.plot(X.numpy(), F.numpy(), label = 'f')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()


Y = F + 0.1 * torch.randn(X.size())
plt.plot(X.numpy(), Y.numpy(), 'r+', label = 'Y')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

LOSS = [] # 记录损失值
ppot = plot_diagram(X,Y,w,stop=5)
for epoch in range(10):
    YHAT = forward(X)
    loss = criterion(YHAT, Y)
    LOSS.append(loss)
    ppot(YHAT, w, loss.item(), epoch)
    loss.backward()
    w.data = w.data - 0.1 * w.grad.data
    w.grad.data.zero_()

plt.plot(LOSS, label = 'loss')
plt.show()