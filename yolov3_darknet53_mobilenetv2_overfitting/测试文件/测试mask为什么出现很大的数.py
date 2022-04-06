import torch

a = torch.tensor([[0,1,2,3,4],
                  [5,6,7,8,9],
                  [10,11,12,13,14]])


mask = a > 10
print(mask)
print(a.shape)

ret = a[mask]
print(ret)