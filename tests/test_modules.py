import pytest
from deeponets import modules 
import torch.nn as nn
import torch

TEST_LN = ['batchnorm', 'layernorm', 'None']
TEST_ACTIVATION = [nn.ReLU(), nn.Tanh(), nn.Identity()]

@pytest.mark.parametrize('ln', TEST_LN)
@pytest.mark.parametrize('activation', TEST_ACTIVATION)
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

@pytest.mark.parametrize('ln', TEST_LN)
@pytest.mark.parametrize('activation', TEST_ACTIVATION)
@pytest.mark.parametrize('final_activation', TEST_ACTIVATION)
@pytest.mark.parametrize('dim_layers', [[1,2,4], [3,5,3]])
def test_resmlp(dim_layers, activation, final_activation, ln):
    mlp = modules.MLP(dim_layers=dim_layers, activation=activation, final_activation=final_activation, ln=ln)

    Nbatch = 10 
    NLength = 20
    #(N,L,C)
    x = torch.randn(Nbatch, NLength, dim_layers[0])
    y = mlp(x)
    assert y.shape == (Nbatch,NLength,dim_layers[-1])
    #(N,C)
    x = torch.randn(Nbatch, dim_layers[0])
    y = mlp(x)
    assert y.shape == (Nbatch, dim_layers[-1])