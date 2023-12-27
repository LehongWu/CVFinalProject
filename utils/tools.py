import torch
import torch.nn.functional as F
import os

class AvgMeter():
    def __init__(self, N=None):
        self.N = N if N is not None else 1000000
        self.cnt = 0
        self.data = []
        self.total = 0.

    def update(self, x):
        self.data.append(x)
        self.total += x
        self.cnt += 1

        if self.cnt > self.N:
            self.total -= self.data[0]
            self.cnt -= 1

    def avg(self):
        return self.total / self.cnt if self.cnt > 0 else 0

