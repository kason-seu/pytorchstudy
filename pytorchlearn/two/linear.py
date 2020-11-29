from torch.nn import Linear
import torch
from LR import LR
lr = Linear(in_features=1, out_features=1, bias=True)

torch.manual_seed(1)

print("lr parameters : ", list(lr.parameters()))

print("Python dictionary ", lr.state_dict())
print("keys ", lr.state_dict().keys())
print("valus ", lr.state_dict().values())

print("weight ", lr.weight)
print("bias ", lr.bias)

x = torch.tensor([[1.0]])
yhat= lr(x)
print("yhat: ", yhat)

x = torch.tensor([[1.0],[3.0]])
yhat = lr(x)
print("yaht: ", yhat)

lr = LR(1,1)
print("python dictionary, ", lr.state_dict())

x = torch.tensor([[1.5],[2.7]])

out = lr.forward(x)
print("forward out = ", out)