import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

class ResidualBlock(nn.Module):
    def __init__(self, dim_in:int, dim_out:int, activation=nn.ReLU(), ln=False):
        """
        `dim_in`: input dimension
        `dim_out`: output dimension
        `activation`: activation function (nn.ReLU() by default)
        `ln`: whether to use layer normalization
        """
        super(ResidualBlock, self).__init__()
        self.layer1 = nn.Linear(dim_in, dim_out)
        self.layer2 = nn.Linear(dim_out, dim_out)
        if ln:
            self.norm1 = nn.BatchNorm1d(dim_in)
            self.norm2 = nn.BatchNorm1d(dim_out)
        else: 
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        self.activation = activation

        if dim_in == dim_out:
            self.layer_in = nn.Identity()
        else:
            self.layer_in = nn.Linear(dim_in, dim_out)
    
    def forward(self, x):
        """
        Clean Path
        """
        # T1
        o = x
        if x.dim() == 3:
            o = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        else:
            o = self.norm1(x)
        o = self.activation(o)
        o = self.layer1(o) 
        # T2 
        if x.dim() == 3:
            o = self.norm2(o.transpose(1, 2)).transpose(1, 2)
        else:
            o = self.norm2(o)
        o = self.activation(o)
        o = self.layer2(o)
        # Residual connection
        x = self.layer_in(x)
        o = x + o
        return o