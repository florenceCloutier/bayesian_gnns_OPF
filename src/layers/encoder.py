import torch
from torch import Tensor
from torch import nn
from typing import Tuple, Union
from collections import OrderedDict

from torch_geometric.nn import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, PairTensor

from .bayesian_linear import BayesianLinear


class GNEncoder(MessagePassing):

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        hidden_size: int,
        aggr: str = "add",
        dropout_rate=0.5,
        use_dropout=False,
        use_va=False,
    ):
        super().__init__(aggr=aggr)

        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.use_dropout = use_dropout
        self.use_va = use_va
        self.dropout_rate = dropout_rate

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        LinearLayer: nn.Module = BayesianLinear if use_va else Linear

        self.edge_mlp = self.edge_mlp = nn.Sequential(
            LinearLayer(in_channels[0], hidden_size), 
            nn.ReLU(), 
            nn.Dropout(p=dropout_rate) if use_dropout else nn.Identity(),
            LinearLayer(hidden_size, hidden_size)
        )
        self.node_mlp = nn.Sequential(
            LinearLayer(in_channels[1], hidden_size), 
            nn.Dropout(p=dropout_rate) if use_dropout else nn.Identity(),
            LinearLayer(hidden_size, hidden_size)
        )
        self.edge_embedding = nn.Parameter(torch.randn(1, hidden_size))
        self.updated_edge_attr = Tensor()

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.edge_mlp:
            for layer in self.edge_mlp.children():
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()
        for layer in self.node_mlp.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the GN encoder.

        Args:
            x: Node features [num_nodes, node_input_size]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_input_size]

        Returns:
            Tuple of:
            - Updated node features [num_nodes, hidden_size]
            - Updated edge features [num_edges, hidden_size]
        """
        if isinstance(x, Tensor):
            x = (x, x)

        if edge_attr is None:
            self.edge_mlp = None

        node_features = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

        return node_features, self.updated_edge_attr

    def message(self, x_j: Tensor, edge_attr: Tensor, edge_index: Adj) -> Tensor:
        """
        Constructs messages from source nodes and edges.
        """
        if edge_attr is None:
            num_edges = edge_index.size(1)
            self.updated_edge_attr = self.edge_embedding.repeat(num_edges, 1)
        else:
            self.updated_edge_attr = self.edge_mlp(edge_attr)
        return self.updated_edge_attr

    def update(self, aggr_out: Tensor, x: PairTensor) -> Tensor:
        """
        Updates node features using the aggregated messages.
        """
        return self.node_mlp(x[1])
    
    def kl_loss(self):
        """Calculate total KL divergence for the module"""
        assert self.use_va, "Trying to get kl_loss when not using va"
        kl_loss_total = 0.0
        if self.edge_mlp:
            for layer in self.edge_mlp.children():
                if hasattr(layer, "kl_loss"):
                    kl_loss_total = kl_loss_total + layer.kl_loss()
        for layer in self.node_mlp.children():
            if hasattr(layer, "kl_loss"):
                kl_loss_total = kl_loss_total + layer.kl_loss()

        return kl_loss_total

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        edge_state_dict = state_dict.get(prefix + 'edge_mlp', None)
        if edge_state_dict:
            self.edge_mlp._load_from_state_dict(edge_state_dict, prefix + 'edge_mlp.', *args, **kwargs)
        else:
            self.edge_mlp = None
        
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
