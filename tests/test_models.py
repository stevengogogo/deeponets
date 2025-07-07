"""
Test DeepONets collection
"""
import torch
from deeponets.models import DeepONet
import pytest 

@pytest.mark.parametrize("deeponet", [DeepONet])
@pytest.mark.parametrize("n_branch", [1, 5])
@pytest.mark.parametrize("n_trunk", [2])
@pytest.mark.parametrize("ln", ['batchnorm', 'layernorm', 'None'])
def test_deeponets(deeponet, n_branch, n_trunk,ln):
    model = deeponet(
        n_branch=n_branch,
        n_trunk=2,
        n_inter=4,
        hiddens_branch=[5, 6],
        hiddens_trunk=[7],
        activation=torch.nn.ReLU(),
        final_activation_branch=torch.nn.Identity(),
        final_activation_trunk=torch.nn.Identity(),
        ln=ln
    )

    n_query = 20
    n_batch = 100
    x_branch = torch.randn(n_batch, n_branch)  # Batch size of 10, branch input dimension of 3
    x_trunk = torch.randn(n_batch, n_query, n_trunk)  # Batch

    gx = model(x_branch, x_trunk)

    assert gx.shape == (n_batch, n_query, 1)  # Output shape should be (batch_size, query_length, 1)