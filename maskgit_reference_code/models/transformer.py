import torch
import torch.nn as nn
import torch.nn.functional as F


class TransfomerLayer(nn.Module):

    def __init__(self, dim, hidden_dim):
        super().__init__()

        # multi-head attention 
        # add & layer normalization
        # feed forward
        # add & layer normalization
        # ......
    
    def forward(self, x):
        pass


class BidirectionalTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # positional embedding
        # transfomer layer
        # read learned codebook 
        # ......

    def forward(self, x):
        pass
