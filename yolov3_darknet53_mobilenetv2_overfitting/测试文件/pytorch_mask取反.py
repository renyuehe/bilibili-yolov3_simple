import torch

mask = torch.tensor([False, False, True])
one = torch.ones(mask.shape).bool()

# print(one - mask)
print(~mask)