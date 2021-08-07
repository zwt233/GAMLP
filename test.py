import torch
a = torch.randn(3, 3)
print(a)
values,indices=torch.max(a,1)
print(values)
print(indices)
