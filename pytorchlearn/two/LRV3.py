# These are the libraries we are going to use in the lab.
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
# The class for plot the diagram

class plot_error_surfaces(object):

    # Constructor
    def __init__(self, w_range, b_range, X, Y, lr, n_samples = 30, go = True):
        W = np.linspace(-w_range, w_range, n_samples)
        B = np.linspace(-b_range, b_range, n_samples)
        self.lr = lr
        w, b = np.meshgrid(W, B)
        Z = np.zeros((30,30))
        count1 = 0
        self.y = Y.numpy()
        self.x = X.numpy()
        for w1, b1 in zip(w, b):
            count2 = 0
            for w2, b2 in zip(w1, b1):
                Z[count1, count2] = np.mean((self.y - w2 * self.x + b2) ** 2)
                count2 += 1
            count1 += 1
        self.Z = Z
        self.w = w
        self.b = b
        self.W = []
        self.B = []
        self.LOSS = []
        self.n = 0
        if go == True:
            plt.figure()
            plt.figure(figsize = (7.5, 5))
            plt.axes(projection='3d').plot_surface(self.w, self.b, self.Z, rstride = 1, cstride = 1,cmap = 'viridis', edgecolor = 'none')
            plt.title('Cost/Total Loss Surface')
            plt.xlabel('w')
            plt.ylabel('b')
            plt.show()
            plt.figure()
            plt.title('Cost/Total Loss Surface Contour')
            plt.xlabel('w')
            plt.ylabel('b')
            plt.contour(self.w, self.b, self.Z)
            plt.show()

    # Setter
    def set_para_loss(self, lr, loss):
        self.n = self.n + 1
        self.W.append(lr.w.data)
        self.B.append(lr.b.data)
        self.LOSS.append(loss)

    # Plot diagram
    def final_plot(self):
        ax = plt.axes(projection = '3d')
        ax.plot_wireframe(self.w, self.b, self.Z)
        ax.scatter(self.W,self.B, self.LOSS, c = 'r', marker = 'x', s = 200, alpha = 1)
        plt.figure()
        plt.contour(self.w,self.b, self.Z)
        plt.scatter(self.W, self.B, c = 'r', marker = 'x')
        plt.xlabel('w')
        plt.ylabel('b')
        plt.show()

    # Plot diagram
    def plot_ps(self):
        plt.subplot(121)
        plt.ylim
        plt.plot(self.x, self.y, 'ro', label="training points")
        plt.plot(self.x, self.W[-1] * self.x + self.B[-1], label = "estimated line")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.ylim((-10, 15))
        plt.title('Data Space Iteration: ' + str(self.n))

        plt.subplot(122)
        plt.contour(self.w, self.b, self.Z)
        plt.scatter(self.W, self.B, c = 'r', marker = 'x')
        plt.title('Total Loss Surface Contour Iteration' + str(self.n))
        plt.xlabel('w')
        plt.ylabel('b')
        plt.show()


class LR(object):
    def __init__(self,w,b,lr):
        self.w = w
        self.b = b
        self.lr = lr

    def foward(self, X):
        return self.w * X + b

    def backward(self, loss):
        loss.backward()

    def update(self,w, b):
        self.w.data = self.w.data - self.lr * w.grad.data
        w.grad.data.zero_()
        self.b.data = self.b.data - self.lr * b.grad.data
        b.grad.data.zero_()

    def cost(self, YHAT,Y):
        return torch.mean((YHAT-Y) ** 2)


if __name__ == '__main__':
    w = torch.tensor(1.5, requires_grad=True)
    b = torch.tensor(-10.0, requires_grad=True)
    X = torch.arange(-3,3,0.1).view(-1,1)
    T = 1 * X - 1
    Y = T + 0.1 * torch.randn(X.size())
    plt.plot(X.numpy(), T.numpy(), '*')
    plt.plot(X.numpy(), Y.numpy(), 'o')
    plt.show()

    lr = LR(w,b,lr=0.1)
    epochs = 15
    LOSS = []
    plot_error_surfaces = plot_error_surfaces(15, 15, X, Y, lr, 30)
    for epoch in range(epochs):
        YHAT = lr.foward(X)
        loss = lr.cost(YHAT, Y)
        LOSS.append(loss)
        plot_error_surfaces.set_para_loss(lr,loss)
        if epoch % 3 == 0:
            plot_error_surfaces.plot_ps()
        lr.backward(loss)
        lr.update(w,b)


    plt.plot(LOSS)
    plt.xlabel("iterations")
    plt.ylabel("Epoch cost")
    plt.show()


