import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

class ResidualBlock(nn.Module):
    def __init__(self, in_channel:int, out_channel:int, activation=nn.ReLU(), ln='batchnorm'):
        """
        `in_channel`: input dimension
        `out_channel`: output dimension
        `activation`: activation function (nn.ReLU() by default)
        `ln`: whether to use layer normalization. This can be batchnorm, layernorm, or None (default: identitiy).
        """
        super(ResidualBlock, self).__init__()
        self.layer1 = nn.Linear(in_channel, out_channel)
        self.layer2 = nn.Linear(out_channel, out_channel)
        self.ln = ln
        if ln=='batchnorm':
            self.norm1 = nn.BatchNorm1d(in_channel)
            self.norm2 = nn.BatchNorm1d(out_channel)
        elif ln=='layernorm':
            self.norm1 = nn.LayerNorm(in_channel)
            self.norm2 = nn.LayerNorm(out_channel)
        else: # Default to no normalization
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        self.activation = activation

        if in_channel == out_channel:
            self.layer_in = nn.Identity()
        else:
            self.layer_in = nn.Linear(in_channel, out_channel)
    
    def forward(self, x):
        """
        Clean Path
        x: input tensor of shape (batch_size, sequence_length, in_channel) or (batch_size, in_channel)
        """
        # T1
        o = x
        if (x.dim() == 3) and (self.ln=='batchnorm'):
            # This is for batchnorm 1D (N,C,L) -> (N,L,C)
            o = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        else:
            o = self.norm1(x)
        o = self.activation(o)
        o = self.layer1(o) 
        # T2 
        if (x.dim() == 3) and (self.ln=='batchnorm'):
            o = self.norm2(o.transpose(1, 2)).transpose(1, 2)
        else:
            o = self.norm2(o)
        o = self.activation(o)
        o = self.layer2(o)
        # Residual connection
        x = self.layer_in(x)
        o = x + o
        return o