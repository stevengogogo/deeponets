import pytest
from deeponets import modules 
import torch.nn as nn
import torch

@pytest.mark.parametrize('ln', ['batchnorm', 'layernorm', 'None'])
@pytest.mark.parametrize('activation', [nn.ReLU(), nn.Tanh(), nn.Identity()])
@pytest.mark.parametrize('out_channel', [1, 5])
@pytest.mark.parametrize('in_channel', [1, 5])
def test_residual_block(in_channel, out_channel, activation, ln):

    block = modules.ResidualBlock(in_channel=in_channel, out_channel=out_channel, activation=activation, ln=ln)

    Nbatch = 10 
    NLength = 20
    # (N, Channel, Length) or (N, Channel)
    # Follow by `BatchNorm1D`: https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
    x = torch.randn(Nbatch, NLength, in_channel)
    y = block(x)
    assert y.shape == (Nbatch,NLength,out_channel)
    x = torch.randn(Nbatch, in_channel)
    y = block(x)
    assert y.shape == (Nbatch, out_channel) 
