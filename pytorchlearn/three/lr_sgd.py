
import torch
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(1)

X = torch.arange(-3,3,0.1).view(-1,1)
f = 1 * X - 1
Y = f + 0.1 * torch.randn(X.size())

print(X)
print(Y)

print(X.size())
print(Y.size())

plt.plot(X.numpy(), Y.numpy(), 'r+', label = 'y')
plt.plot(X.numpy(), f.numpy(), label = 'f')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

for x,y in zip(X,Y):
    print('x = ',x.data, 'y = ', y.data)



# 定义正向传播，反向传播
def forward(w,x,b):
    return w * x + b

def criterion(yhat,y):
    return torch.mean((yhat-y)**2)

from base.plot_error_surfaces import plot_error_surfaces
get_surface = plot_error_surfaces(15, 13, X, Y, 30)

w = torch.tensor(-15.0, requires_grad=True)
b = torch.tensor(-10.0, requires_grad = True)

epochs = 10
lr = 0.1
# batch gradient descent
LOSS_BGD = []

for epoch in range(epochs):
    YHAT = forward(w,X,b)
    loss = criterion(YHAT,Y)
    get_surface.set_para_loss(w.data.tolist(), b.data.tolist(), loss.tolist())
    get_surface.plot_ps()
    LOSS_BGD.append(loss)
    loss.backward()
    w.data = w.data - lr * w.grad.data
    b.data = b.data - lr * b.grad.data

    w.grad.data.zero_()
    b.grad.data.zero_()


w = torch.tensor(-15.0, requires_grad=True)
b = torch.tensor(-10.0, requires_grad = True)

epochs = 10
lr = 0.1
#  Stochastic Gradient Descent
LOSS_SGD = []

for epoch in range(epochs):
    YHAT = forward(w,X,b)
    LOSS_SGD.append(criterion(YHAT,Y).tolist())
    for x, y in zip(X,Y):
        # make a pridiction
        yhat = forward(w,x,b)

        # calculate the loss
        loss = criterion(yhat, y)

        # Section for plotting
        get_surface.set_para_loss(w.data.tolist(), b.data.tolist(), loss.tolist())

        # backward pass: compute gradient of the loss with respect to all the learnable parameters
        loss.backward()

        # update parameters slope and bias
        w.data = w.data - lr * w.grad.data
        b.data = b.data - lr * b.grad.data

        # zero the gradients before running the backward pass
        w.grad.data.zero_()
        b.grad.data.zero_()
    # plot after each epoch
    get_surface.plot_ps()


plt.plot(LOSS_BGD,'r+',label = 'Batch Gradient Descent')
plt.plot(LOSS_SGD, 'go', label = 'Stochastic Gradient Descent')
plt.xlabel('epoch')
plt.ylabel('cost / total loss')
plt.legend()
plt.show()


