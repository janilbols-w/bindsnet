from bindsnet.encoding import poisson
import matplotlib.pyplot as plt
import torch

plt.matshow(poisson(torch.tensor(range(100)).float()*100., 200))
plt.show()
