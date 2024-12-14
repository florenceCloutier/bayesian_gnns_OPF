import torch
import torch.nn.functional as F
from torch_geometric.nn import to_hetero

from layers.bayesian_graph_conv import BayesianGraphConv

# Custom to_hetero wrapper
def custom_to_hetero(model, metadata):
    hetero_model = to_hetero(model, metadata)
    return HeteroBayesianGNN(hetero_model, model.hidden_channels, model.in_channels, model.out_channels, model.num_layers)


class BayesianGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()

        self.convs.append(BayesianGraphConv(in_channels, hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(BayesianGraphConv(hidden_channels, hidden_channels))

        self.convs.append(BayesianGraphConv(hidden_channels, out_channels))

    def forward(self, x, edge_index, edge_attr_dict=None):
        for i in range(len(self.convs) - 1):
            conv = self.convs[i]
            x = conv(x, edge_index)
            x = F.relu(x)

        x = self.convs[-1](x, edge_index)
        return x

    def kl_loss(self):
        """Calculate total KL loss for all layers"""
        return sum(conv.kl_loss() for conv in self.convs)


class HeteroBayesianGNN(torch.nn.Module):
    def __init__(self, base_model, in_channels, hidden_channels, out_channels, num_layers=2):
        super().__init__()
        self.model = base_model
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def kl_loss(self):
        kl_loss = 0
        for module in self.model.modules():
            if isinstance(module, BayesianGraphConv):
                kl_loss += module.kl_loss()
        return kl_loss
    
    def get_init_kwargs(self):
        """Return initialization arguments for checkpointing."""
        return {
            'in_channels': self.in_channels,
            'hidden_channels': self.hidden_channels,
            'out_channels': self.out_channels,
            'num_layers': self.num_layers,
        }