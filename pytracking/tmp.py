import torch

scores = torch.rand([1,19,19])
core_mask = torch.zeros(scores.size(),dtype=scores.dtype)
core_mask[0,8:11,8:11] = 1
peri_mask = 1 - core_mask
print(peri_mask)