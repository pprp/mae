import numpy as np

import torch

a = [np.random.randint(0, 24) for _ in range(24)]

a = np.array(a).reshape(2, 3, 4)

a = torch.tensor(a, dtype=torch.float32)

print("a:", a)

mean = torch.mean(a, dim=2)

idx = torch.argsort(mean, dim=1, descending=True)

print("mean:", mean)

print("idx:", idx)

print("mean after sort:", torch.gather(mean, dim=1, index=idx))

mask = torch.ones([2, 3])

mask[:, :2] = 0

print("after set zero mask:", mask)

mask = torch.gather(mask, dim=1, index=idx)

print("after gather mask:", mask)

print("actual masking: ", (1-mask.unsqueeze(-1).repeat(1, 1, 4)))

print("after masking a:", a * (1-mask.unsqueeze(-1).repeat(1, 1, 4)))
