"""
Collection of DeepONets
"""
import torch 
import torch.nn as nn
from functorch import make_functional
from .modules import MLP
import warnings
from functorch.experimental import replace_all_batch_norm_modules_

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

class NOMDADDeepONet(DeepONetBase):
    """
    DeepONet with NOMDAD (Nonlinear Manifold Decoders for Operator Learning)
    Ref: 
    1. Paper: https://openreview.net/pdf?id=5OWV-sZvMl
    2. Code: https://github.com/PredictiveIntelligenceLab/NOMAD/blob/86ce2049dcc7446c356563f168f191c14f147e25/Antiderivative/Train_model/train_antiderivative.py#L112-L119
    """
    def __init__(self, n_branch:int, n_trunk:int, n_inter:int, hiddens_branch:list[int]=[], hiddens_trunk:list[int]=[], activation=nn.ReLU(),final_activation_branch=nn.Identity(), final_activation_trunk=nn.Identity(), ln=None, n_output=1):
        super().__init__(n_branch, n_trunk, n_inter, n_output=n_output)
        self.n_layers_branch = [n_branch, *hiddens_branch, n_inter]
        self.n_layers_trunk = [n_inter + n_trunk, *hiddens_trunk, n_output]

        self.branch = MLP(self.n_layers_branch, activation=activation, final_activation=final_activation_branch, ln=ln)
        self.trunk = MLP(self.n_layers_trunk, activation=activation, final_activation=final_activation_trunk, ln=ln)
        
        # Batchnorm case is patched
        if ln=="batchnorm":
            warnings.warn("NOMADDeepONet does not support batch normalization because of vmap. Using layer normalization instead. Patch all batchnorm via https://docs.pytorch.org/functorch/stable/batch_norm.html")
            self.trunk = replace_all_batch_norm_modules_(self.trunk)


    def forward(self, x_branch:torch.Tensor, x_trunk:torch.Tensor):
        """
        x_branch: (n_batch, n_branch)
        y_trunk: (n_batch, n_query, n_trunk)
        """
        if x_trunk.dim() == 2:
            # If x_trunk is (n_batch, n_trunk), we need to unsqueeze it to (n_batch, 1, n_trunk). means 1 example
            x_trunk = x_trunk.view(x_trunk.shape[0], 1, x_trunk.shape[1])

        def nomad(br, tr):
            """
            br: branch input (n_batch, n_branch_out)
            tr: trunk input (n_batch, n_trunk)
            """
            inp = torch.cat([br, tr], dim=-1) # (n_batch, n_branch_out + n_trunk)
            return self.trunk(inp) # (n_batch, n_query, n_output)
        br = self.branch(x_branch)  # (n_batch, n_branch_out)
        out = torch.vmap(nomad, in_dims=(None, 1))(br, x_trunk) # (n_query, n_batch, n_output)
        out = out.transpose(0,1)  # (n_batch, n_query, n_output)
        return out#.transpose(2,1)