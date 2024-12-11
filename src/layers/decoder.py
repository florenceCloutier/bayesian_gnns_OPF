import torch
from torch import Tensor
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import OptPairTensor, Adj, PairTensor
from typing import Union


class GNDecoder(MessagePassing):
    """
    Graph Network Decoder that projects latent features to output values for
    bus voltage and generator power.
    """

    def __init__(self, 
                 hidden_size: int, 
                 out_channels: int,
                 dropout_rate: float = 0.5,
                 use_dropout: bool = False):
        super().__init__(aggr="add")

        self.out_channels = out_channels
        self.hidden_size = hidden_size

        self.edge_mlp = nn.Sequential(
            Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate) if use_dropout else nn.Identity(),
            nn.LayerNorm(256),
            Linear(256, 256, bias=False),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate) if use_dropout else nn.Identity(),
            nn.LayerNorm(256),
        )
        self.node_mlp = nn.Sequential(
            Linear(hidden_size+256, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate) if use_dropout else nn.Identity(),
            nn.LayerNorm(256),
            Linear(256, 256, bias=False),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate) if use_dropout else nn.Identity(),
            nn.LayerNorm(256),
        )
        self.output_layer = Linear(256, self.out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        for layer in self.edge_mlp.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.node_mlp.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.output_layer.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr: Tensor) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)
            
        node_features = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)
        output = self.output_layer(node_features)

        return output, None

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return self.edge_mlp(edge_attr)

    def update(self, aggr_out: Tensor, x: PairTensor) -> Tensor:
        combined = torch.cat([x[1], aggr_out], dim=-1)
        return self.node_mlp(combined)
