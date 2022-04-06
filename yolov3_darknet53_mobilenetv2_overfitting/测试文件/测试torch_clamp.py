# 将输入input张量每个元素的夹紧到区间 [min,max][min,max]，并返回结果到一个新张量。

import torch

a=torch.randint(low=0,high=10,size=(10,1))
print(a)

b=torch.clamp(a,3,9)
print(b)

c = torch.clamp(a,min=3)
print(c)

