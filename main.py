import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import numpy as np


a = np.array([1, 2, 3, 4, 5])
b = a[:-2] * a[1:-1] * a[2:]
print(b)
