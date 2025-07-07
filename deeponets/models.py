"""
Collection of DeepONets
"""
import torch 
import torch.nn as nn
from functorch import make_functional
from .modules import MLP

class DeepONetBase(nn.Module):
    def __init__(self, n_branch, n_trunk, n_inter, n_output):
        """
        Base class for DeepONet. This is an abstract class and should not be used directly.
        `n_branch`: input dimension of branch net
        `n_trunk`: input dimension of trunk net
        `n_inter`: output dimension of branch net and trunk net
        `n_output`: output dimension of DeepONet
        """
        super().__init__()
        self.n_branch = n_branch 
        self.n_trunk = n_trunk
        self.n_inter = n_inter
        self.n_output = n_output
    def forward(self, x_branch:torch.Tensor, x_trunk:torch.Tensor):
        """
        x_branch: (n_batch, n_branch)
        y_trunk: (n_batch, n_query, n_trunk)
        """
        raise NotImplementedError("DeepONet is not implemented yet. Please use UnstackDeepONet instead.")

class DeepONet(DeepONetBase):
    """
    Vanilla DeepONet 
    Ref: 
    """
    def __init__(self, n_branch:int, n_trunk:int, n_inter:int, hiddens_branch:list[int]=[], hiddens_trunk:list[int]=[], activation=nn.ReLU(),final_activation_branch=nn.Identity(), final_activation_trunk=nn.Identity(), ln=False):
        """
        DeepONet for operator learning. output is 1 channel.
        `n_branch`: input dimension of branch net
        `n_trunk`: input dimension of trunk net
        `n_inter`: output dimension of branch net and trunk net
        `n_hidden`: hidden dimension of branch net and trunk net
        `n_layer`: number of layers in branch net and trunk net
        `hiddens_branch`: list of hidden dimensions for branch net
        `hiddens_trunk`: list of hidden dimensions for trunk net
        `activation`: activation function for branch net and trunk net (default: nn.ReLU())
        `final_activation_branch`: final activation function for branch net (default: nn.Identity())
        `final_activation_trunk`: final activation function for trunk net (default: nn.Identity())
        `ln`: whether to use layer normalization
        """
        super().__init__(n_branch, n_trunk, n_inter, n_output=1)
        self.n_layers_branch = [n_branch, *hiddens_branch, n_inter]
        self.n_layers_trunk = [n_trunk, *hiddens_trunk, n_inter]
       

        self.branch = MLP(self.n_layers_branch, activation=activation, final_activation=final_activation_branch, ln=ln)
        self.trunk = MLP(self.n_layers_trunk, activation=activation, final_activation=final_activation_trunk, ln=ln)
        self.bias = nn.Parameter(torch.randn(1,), requires_grad=True)
    
    def forward(self, x_branch, x_trunk):
        """
        x_branch: (n_batch, n_branch)
        y_trunk: (n_batch, n_query, n_trunk)
        """
        if x_trunk.dim() == 2:
            # If x_trunk is (n_batch, n_trunk), we need to unsqueeze it to (n_batch, 1, n_trunk). means 1 example
            x_trunk = x_trunk.view(x_trunk.shape[0], 1, x_trunk.shape[1])

        branch_out = self.branch(x_branch) # (n_batch, n_inter)
        trunk_out = self.trunk(x_trunk) # (n_batch, n_query, n_inter)
        branch_out = branch_out.unsqueeze(-1) # (n_batch, n_inter, 1)
        # Inner product (batched)
        out = torch.bmm(trunk_out, branch_out) + self.bias # (n_batch, n_query, 1)
        return out

