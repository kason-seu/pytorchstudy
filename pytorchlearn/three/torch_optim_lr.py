import numpy as np

import torch

from torch import nn
from torch.utils.data import Dataset,DataLoader
from torch import optim
class LR(nn.Module):

    def __init__(self, inputs, outputs):
        super(LR, self).__init__()
        self.lr = nn.Linear(inputs, outputs)

    def forward(self, x):
        yhat = self.lr.forward(x)
        return yhat


class LRData(Dataset):
    def __init__(self, w = 3, b = 0.1):
        self.x = torch.arange(-3,3,0.1).view(-1,1)
        self.y = self.x * w - b + 0.1 * torch.randn(self.x.size())

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index],self.y[index]

data = LRData()

dataloader = DataLoader(dataset=data, batch_size=5)
# for x,y in dataloader:
#     print(x,y)

print("iterate second")
# for x, y in enumerate(data):
#     print(x,y)



epochs = 10
lr = LR(1,1)
lr.state_dict()['lr.weight'][0] = -15
lr.state_dict()['lr.bias'][0] = 10
print(lr.state_dict())


optim =optim.SGD(params = lr.parameters(), lr= 0.01)

LOSS = []

cost = nn.MSELoss()
for epoch in range(epochs):

    for x,y in dataloader:
        yhat = lr(x)
        loss = cost(yhat,y)
        LOSS.append(loss)
        optim.zero_grad()
        loss.backward()
        optim.step()

import matplotlib.pyplot as plt

print(len(LOSS))
plt.plot(LOSS, 'g+', label = 'cost')
plt.xlabel('epoch')
plt.ylabel('cost')
plt.legend()
plt.show()



