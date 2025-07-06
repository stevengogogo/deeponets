"""
Test DeepONets collection
"""
import torch
from deeponets.models import UnstackDeepONet
import pytest 

@pytest.mark.parametrize("ln", ['batchnorm', 'layernorm', 'None'])
def test_deeponets(ln):
    model = UnstackDeepONet(
        n_branch=3,
        n_trunk=2,
        n_inter=4,
        hiddens_branch=[5, 6],
        hiddens_trunk=[7],
        activation=torch.nn.ReLU(),
        final_activation_branch=torch.nn.Identity(),
        final_activation_trunk=torch.nn.Identity(),
        ln=True
    )

    x_branch = torch.randn(10, 3)  # Batch size of 10, branch input dimension of 3
    x_trunk = torch.randn(10, 20, 2)  # Batch

    gx = model(x_branch, x_trunk)

    assert gx.shape == (10, 20, 1)  # Output shape should be (batch_size, query_length, 1)