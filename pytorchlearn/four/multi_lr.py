import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim
import matplotlib.pyplot as plt

class LR(nn.Module):
    def __init__(self, inputs_features, output_features):
        super(LR, self).__init__()
        self.lr = nn.Linear(inputs_features, output_features)

    def forward(self, x):
        return self.lr(x)


lr = LR(2,1)
print(list(lr.parameters()))
print(dict(lr.state_dict()))

yhat = lr(torch.tensor([[0.1,2]]))
print(yhat)

yhat2 = lr(torch.tensor([[0.1,2],[0.2,3],[0.5,1.2]]))
print(yhat2)

x =  torch.arange(-3, 3, 0.1).view(-1,2)
print(x)
y = x.mm(torch.tensor([[-1.0],[3.0]]))
print(y)
class Data(Dataset):
    def __init__(self):
        self.x = torch.arange(-3, 3, 0.1).view(-1,2)
        self.f = self.x.mm(torch.tensor([[-1.0],[3.0]])) - 10
        self.y = self.f + 0.001 * torch.randn(len(self.x))
        self.len = self.x.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.x[item], self.y[item]

data = Data()
print(data.x.shape[1])
epochs = 1000
lr = LR(data.x.shape[1], data.y.shape[1])
lr.state_dict()['lr.weight'] = torch.tensor([-2.0,5.0])
lr.state_dict()['lr.bias'] = torch.tensor([-10.0])
print('parameters = ', dict(lr.state_dict()))
criterion = nn.MSELoss()
opt = optim.SGD(params=lr.parameters(), lr = 0.1)
LOSS = []
for epoch in range(epochs):

    yhat = lr(data.x)
    loss = criterion(yhat, data.y)
    LOSS.append(loss)
    opt.zero_grad()
    loss.backward()
    opt.step()



plt.plot(LOSS,'*')
plt.show()

print("-------------------------------------")
print(list(lr.parameters()))

yhat_result = lr(data.x)

plt.plot(data.y.numpy(), 'b*')
plt.plot(yhat_result.detach().numpy())
plt.show()