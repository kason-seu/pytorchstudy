import torch
### 梯度下降with dataset
from torch.utils.data import Dataset,DataLoader
class Data(Dataset):
    def __init__(self):
        self.x = torch.arange(-3,3,0.1).view(-1,1)
        self.y = 1 * self.x - 1 + torch.randn(1)
        self.len = self.x.shape[0]

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return self.len


dataset = Data()

x,y = dataset[0]
print("(",x,y,")")

x, y = dataset[0:3]
print("(",x,y,")")


loader = DataLoader(dataset=dataset, batch_size = 1)

for x,y in loader:
    print("x = ", x, ", y = ", y)

