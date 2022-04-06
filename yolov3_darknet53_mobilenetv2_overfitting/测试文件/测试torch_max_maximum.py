
import torch
cx = torch.tensor([12])
cxs = torch.tensor([10,11,13])
w = torch.tensor(23)
ws = torch.tensor([20,22,24])

xx1 = torch.maximum(cx - ws /2, cxs - ws /2)
print(xx1)
xx1 = torch.max(cx - ws /2, cxs - ws /2)
print(xx1)

xx1 = torch.maximum(cx - w /2, cxs - ws /2)
print(xx1)
xx1 = torch.max(cx - w /2, cxs - ws /2)
print(xx1)