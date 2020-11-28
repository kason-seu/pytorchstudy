import torch
from torch.utils.data import Dataset
from torchvision import transforms

class toy_set(Dataset):

    def __init__(self, length = 100, transform = None):
        self.len = length

        self.x = 2 * torch.ones(length,2)
        self.y = torch.ones(length, 1)
        self.transform = transform

    def __getitem__(self, item) ->tuple:
        sample = self.x[item], self.y[item]
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.len

class add_mul(object):
    def __init__(self, addx = 1, multy = 2):
        self.addx = addx
        self.multy = multy

    def __call__(self, sample) ->tuple:
        x = sample[0]
        y = sample[1]
        x = x + self.addx
        y = y * self.multy
        sample = x,y
        return sample

class mult2(object):
    def __init__(self, multi=100):
        self.multi = multi

    def __call__(self, sample) -> tuple:
        x = sample[0]
        y = sample[1]
        x = x * self.multi
        y = y * self.multi
        return x,y

if __name__ == '__main__':
    our_dataset = toy_set()
    print("Our toy_set object: ", our_dataset)
    print("Value on index 0 of our toy_set object: ", our_dataset[0])
    print("Our toy_set length: ", len(our_dataset))
    for i in range(3):
        xi,yi = our_dataset[i]
        print("index : ", i , "; x :", xi, "; y:", yi)

    for i,v in our_dataset:
        print("x ", i, " y ", v)

    a_m = add_mul()
    for sample in our_dataset:
        print(a_m(sample))


    print("---------------------------")
    out_dataset_2 = toy_set(100, transform=a_m)
    for sample in out_dataset_2:
        print(sample)

    data_transform = transforms.Compose([add_mul(),mult2()])
    print("before compose ", our_dataset[0])
    print("after compose: ", data_transform(our_dataset[0]))

    out_dataset_3 = toy_set(transform=data_transform)
    for i in range(3):
        print(out_dataset_3[i])




