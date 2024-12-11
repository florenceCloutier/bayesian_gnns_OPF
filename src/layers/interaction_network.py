import torch
from torch import Tensor
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, OptPairTensor

from typing import Union


class RelationalModel(nn.Module):
    def __init__(self, 
                 in_size: int, 
                 hidden_size: int,
                 dropout_rate: float = 0.5,
                 use_dropout: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate) if use_dropout else nn.Identity(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate) if use_dropout else nn.Identity(),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ObjectModel(nn.Module):
    def __init__(self, 
                 in_size: int, 
                 hidden_size: int,
                 dropout_rate: float = 0.5,
                 use_dropout: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate) if use_dropout else nn.Identity(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate) if use_dropout else nn.Identity(),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class InteractionNetworkBlock(MessagePassing):
    """
    IN Block based on the paper "Interaction Networks for Learning about Objects, Relations and Physics"
    """

    def __init__(self, hidden_size: int, dropout_rate: float = 0.5, use_dropout: bool = False):
        super().__init__(aggr="add")

        self.hidden_size = hidden_size

        self.edge_mlp = RelationalModel(hidden_size * 3, hidden_size, dropout_rate, use_dropout)
        self.node_mlp = ObjectModel(hidden_size * 2, hidden_size, dropout_rate, use_dropout)
        self.updated_edge_attr = Tensor()

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        for layer in self.edge_mlp.net.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.node_mlp.net.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr: Tensor) -> tuple[Tensor, Tensor]:
        if isinstance(x, Tensor):
            x = (x, x)

        updated_nodes = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)
        return updated_nodes, self.updated_edge_attr

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        edge_features = torch.cat([x_i, x_j, edge_attr], dim=-1)
        self.updated_edge_attr = edge_attr + self.edge_mlp(edge_features) # Residual connection
        return self.updated_edge_attr 

    def update(self, aggr_out: Tensor, x: Tensor) -> Tensor:
        combined = torch.cat([x[1], aggr_out], dim=-1)
        return x[1] + self.node_mlp(combined) # Residual connection
