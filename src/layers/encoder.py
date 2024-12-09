import torch
from torch import Tensor
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from typing import Tuple, Union, Final
from torch_geometric.typing import Adj, OptPairTensor, PairTensor
from torch_geometric.utils import spmm


class GNEncoder(MessagePassing):
    # SUPPORTS_FUSED_EDGE_INDEX: Final[bool] = True

    def __init__(
        self, in_channels: Union[int, Tuple[int, int]], hidden_size: int, aggr: str = "add"
    ):
        super().__init__(aggr=aggr)

        self.in_channels = in_channels
        self.hidden_size = hidden_size

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.edge_mlp = nn.Sequential(
            Linear(in_channels[0], hidden_size), nn.ReLU(), Linear(hidden_size, hidden_size)
        )
        self.node_mlp = nn.Sequential(
            Linear(-1, hidden_size), nn.ReLU(), Linear(hidden_size, hidden_size)
        )
        self.edge_embedding = nn.Parameter(torch.randn(1, hidden_size))
        self.updated_edge_attr = Tensor()

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        for layer in self.edge_mlp.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.node_mlp.children():
            if hasattr(layer, 'reset_parameters'):
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

    # def message_and_aggregate(self, edge_index: Adj, x: OptPairTensor, edge_attr: Tensor) -> Tensor:
    #     return spmm(edge_index, edge_attr, reduce=self.aggr)

    def update(self, aggr_out: Tensor, x: PairTensor) -> Tensor:
        """
        Updates node features using the aggregated messages.
        """
        combined = torch.cat([x[1], aggr_out], dim=-1)
        return self.node_mlp(combined)
