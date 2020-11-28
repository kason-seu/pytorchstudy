

import torch

a = torch.tensor([7,6,5,4,3,2.0])

print(a.dtype)
print(a)

a = a.type(torch.float64)
print(a.dtype)

a_col = a.view(-1, 1)
print(a_col)
print(a_col.ndimension())

# tensor, numpy, list

# indexing and slices
c = torch.tensor([1,2,3])
c[0] = 100
print(c)
d = c[1:3]
print(d)