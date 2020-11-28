import torch

x = torch.rand(1)
b = torch.rand(1, requires_grad=True)
w = torch.rand(1, requires_grad=True)
y = w * x
z = y + b

print(x.requires_grad, b.requires_grad, w.requires_grad,y.requires_grad,  z.requires_grad)

# 是否是叶子
print(x.is_leaf, b.is_leaf, w.is_leaf, y.is_leaf, z.is_leaf)

z.backward(retain_graph=True)

print(w.grad)
print(x.grad)
print(b.grad)
print(b.grad)
print(b.grad)