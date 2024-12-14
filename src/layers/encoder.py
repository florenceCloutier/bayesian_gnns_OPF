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
        dropout_rate = 0.5,
        use_dropout = False,
        use_va = False,
    ):
        super().__init__(aggr=aggr)

        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.use_dropout = use_dropout
        self.use_va = use_va
        self.dropout_rate = dropout_rate

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.LinearLayer: nn.Module = BayesianLinear if use_va else Linear

        self.edge_mlp = None
        self.node_mlp = nn.Sequential(
            self.LinearLayer(-1, hidden_size), 
            nn.ReLU(), 
            nn.Dropout(p=dropout_rate) if use_dropout else nn.Identity(),
            self.LinearLayer(hidden_size, hidden_size)
        )
        self.edge_embedding = nn.Parameter(torch.randn(1, hidden_size))
        self.updated_edge_attr = Tensor()

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.edge_mlp:
            for layer in self.edge_mlp.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        for layer in self.node_mlp.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    
    def _initialize_edge_mlp(self, edge_attr):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.edge_mlp = nn.Sequential(
                self.LinearLayer(edge_attr.shape[-1], self.hidden_size), 
                nn.ReLU(), 
                nn.Dropout(p=self.dropout_rate) if self.use_dropout else nn.Identity(),
                self.LinearLayer(self.hidden_size, self.hidden_size)
            ).to(device)

        # Load lazy layers
        with torch.no_grad():
            _ = self.edge_mlp(edge_attr)

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
        
        if edge_attr is not None and self.edge_mlp is None:
            self._initialize_edge_mlp(edge_attr)

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
        combined = torch.cat([x[1], aggr_out], dim=-1)
        return self.node_mlp(combined)
    
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

    def state_dict(self, *args, **kwargs):
        """Custom state_dict to handle uninitialized edge_mlp"""
        # Create an empty state dict
        state = OrderedDict()
        
        # Add parameters from node_mlp if initialized
        try:
            node_state = self.node_mlp.state_dict(*args, **kwargs)
            for key, value in node_state.items():
                state[f'node_mlp.{key}'] = value
        except Exception as e:
            print(f"Warning: Could not get state dict for node_mlp: {e}")
            
        # Add parameters from edge_mlp only if it exists and is initialized
        if self.edge_mlp is not None:
            try:
                edge_state = self.edge_mlp.state_dict(*args, **kwargs)
                for key, value in edge_state.items():
                    state[f'edge_mlp.{key}'] = value
            except Exception as e:
                print(f"Warning: Could not get state dict for edge_mlp: {e}")
                
        return state

    def load_state_dict(self, state_dict, strict=True):
        """Custom load_state_dict to handle missing edge_mlp parameters"""
        if strict:
            # If edge_mlp parameters are missing, check if edge_mlp is None
            missing_keys = set(self.state_dict().keys()) - set(state_dict.keys())
            if missing_keys and not all(k.startswith('edge_mlp') for k in missing_keys):
                raise RuntimeError(f"Missing key(s) in state_dict: {missing_keys}")
        return super().load_state_dict(state_dict, strict=False)
    