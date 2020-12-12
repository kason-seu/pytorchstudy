import numpy as np

import torch
import matplotlib.pyplot as plt
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
    def __init__(self, w = 3, b = 0.1, training = True):
        self.x = torch.arange(-3,3,0.1).view(-1,1)
        self.y = self.x * w - b + 0.1 * torch.randn(self.x.size())
        if training:
            self.y[0] = 0
            self.y[55:60] = 20
        else:
            pass
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index],self.y[index]

traing_data = LRData()
valid_data = LRData(training=False)

dataloader = DataLoader(dataset=traing_data, batch_size=1)
# for x,y in dataloader:
#     print(x,y)

# for x, y in enumerate(data):
#     print(x,y)



epochs = 10
learning_rates = [0.0001, 0.001,0.01,0.1]
traing_loss = torch.zeros(len(learning_rates))
valid_loss = torch.zeros(len(learning_rates))
MODELS = []
for i,learning_rate in enumerate(learning_rates):
    lr = LR(1,1)
    lr.state_dict()['lr.weight'][0] = -15
    lr.state_dict()['lr.bias'][0] = 10
    print(lr.state_dict())
    op = optim.SGD(params = lr.parameters(), lr = learning_rate)
    cost = nn.MSELoss()
    LOSS = []
    for epoch in range(epochs):
        for x,y in dataloader:
            yhat = lr(x)
            loss = cost(yhat,y)
            LOSS.append(loss)
            op.zero_grad()
            loss.backward()
            op.step()
    MODELS.append(lr)
    traing_loss[i] = cost(lr(traing_data.x), traing_data.y).item()
    valid_loss[i] = cost(lr(valid_data.x), valid_data.y).item()
    print(len(LOSS))
    plt.plot(LOSS, 'g+', label = 'cost')
    plt.xlabel('epoch')
    plt.ylabel('cost')
    plt.legend()
    plt.show()

plt.semilogx(np.array(learning_rates), traing_loss.numpy(), 'ro', label = 'training loss/total Loss')
plt.semilogx(np.array(learning_rates), valid_loss.numpy(), 'b', label = 'validation cost/total Loss')
plt.legend()
plt.show()


for model, learning_rate in zip(MODELS, learning_rates):
    yhat = model(valid_data.x)
    plt.plot(valid_data.x.numpy(), yhat.detach().numpy())
plt.plot(valid_data.x.numpy(), valid_data.y.numpy(), '*')
plt.show()





