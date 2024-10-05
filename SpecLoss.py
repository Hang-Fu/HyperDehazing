import torch
import numpy as np

class spec_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x1 = x[0, :, :, :]
        M = torch.mean(torch.mean(x1, 1), 1)
        N = torch.mean(torch.mean(y[0, :, :, :], 1), 1)
        return torch.abs(torch.tensor(M - N)).mean()