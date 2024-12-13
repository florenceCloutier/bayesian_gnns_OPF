import torch
from torch import Tensor
import torch.nn as nn
from typing import Union, Tuple, Dict

from torch_geometric.typing import Metadata

from layers.encoder import GNEncoder
from layers.processor import ProcessorModule
from layers.decoder import GNDecoder


def to_hetero_with_edges(model: nn.Module, metadata: Metadata) -> nn.Module:
    """
    Custom wrapper to convert a homogeneous GNN to a heterogeneous GNN that handles both
    node and edge feature outputs.

    Args:
        model: Base homogeneous model that returns (node_features, edge_features)
        metadata: Graph metadata (node_types, edge_types, graph_name)
        aggr: Currently only support sum aggr
    """

    class HeteroWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model
            node_types, edge_types = metadata

            # Create dict of homogeneous models for each edge type
            self.convs = nn.ModuleDict()
            for edge_type in edge_types:
                self.convs[str(edge_type)] = model.__class__(*model.__init_args__, **model.__init_kwargs__)

        def forward(
            self,
            x_dict: Dict[str, Tensor],
            edge_index_dict: Dict[str, Tensor],
            edge_attr_dict: Dict[str, Tensor],
        ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
            # Initialize dictionaries for collecting results
            node_features_dict = {node_type: [] for node_type in x_dict.keys()}
            edge_features_dict = {edge_type: None for edge_type in edge_index_dict.keys()}

            # Process each edge type
            for edge_type, edge_index in edge_index_dict.items():
                src_type, _, dst_type = edge_type
                edge_attr = edge_attr_dict.get(edge_type)

                # Apply the corresponding model
                node_features, edge_features = self.convs[str(edge_type)](
                    (x_dict[src_type], x_dict[dst_type]), edge_index, edge_attr
                )

                # Collect results
                node_features_dict[dst_type].append(node_features)
                edge_features_dict[edge_type] = edge_features

            # Aggregate node features from different edge types
            out_dict = {
                node_type: (
                    torch.stack(features).sum(dim=0)
                    if len(features) > 1
                    else features[0] if len(features) == 1 else torch.zeros_like(x_dict[node_type])
                )
                for node_type, features in node_features_dict.items()
            }

            return out_dict, edge_features_dict

        def reset_parameters(self):
            for conv in self.convs.values():
                conv.reset_parameters()
        
        def kl_loss(self):
            kl_loss = 0
            for conv in self.convs.values():
                kl_loss += conv.kl_loss()
            return kl_loss
        
        def get_init_kwargs(self):
            return model.__init_kwargs__

    # Store initialization arguments
    init_args = tuple()
    init_kwargs = {}
    for param in model.__init__.__code__.co_varnames[1 : model.__init__.__code__.co_argcount]:
        if hasattr(model, param):
            init_kwargs[param] = getattr(model, param)

    model.__init_args__ = init_args
    model.__init_kwargs__ = init_kwargs

    return HeteroWrapper()


class CANOS(torch.nn.Module):
    """
    Complete CANOS architecture with encode-process-decode structure
    """

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        hidden_size: int,
        out_channels: int,
        num_message_passing_steps: int,
        metadata: Metadata,
        dropout_rate: float = 0.5,
        use_dropout: bool = False,
        use_va: bool = False,
    ):
        super().__init__()
        
        # Store initialization arguments
        init_args = tuple()
        init_kwargs = {}
        for param in self.__init__.__code__.co_varnames[1 : self.__init__.__code__.co_argcount]:
            if hasattr(self, param):
                init_kwargs[param] = getattr(self, param)

        self.__init_args__ = init_args
        self.__init_kwargs__ = init_kwargs

        assert not (use_dropout and use_va), "trying to use variational inference with dropout"
        self.use_va = use_va

        self.encoder = to_hetero_with_edges(
            GNEncoder(
                in_channels=in_channels,
                hidden_size=hidden_size,
                dropout_rate=dropout_rate,
                use_dropout=use_dropout,
                use_va=use_va,
            ),
            metadata,
        )
        self.processor = ProcessorModule(
            hidden_size=hidden_size,
            num_message_passing_steps=num_message_passing_steps,
            to_hetero=to_hetero_with_edges,
            metadata=metadata,
            dropout_rate=dropout_rate,
            use_dropout=use_dropout,
            use_va=use_va,
        )
        self.decoder = to_hetero_with_edges(
            GNDecoder(
                hidden_size=hidden_size,
                out_channels=out_channels,
                dropout_rate=dropout_rate,
                use_dropout=use_dropout,
                use_va=use_va,
            ),
            metadata,
        )

    def forward(self, x_dict: Dict[str, Tensor], edge_index_dict: Dict[str, Tensor], edge_attr_dict: Dict[str, Tensor]):
        # Encode
        latent_nodes, latent_edge_attr = self.encoder(x_dict, edge_index_dict, edge_attr_dict)

        # Process
        latent_nodes, latent_edge_attr = self.processor(latent_nodes, edge_index_dict, latent_edge_attr)

        # Decode
        node_features, _ = self.decoder(latent_nodes, edge_index_dict, latent_edge_attr)

        return node_features
        
    def kl_loss(self):
        """Calculate total KL divergence for the module"""
        assert self.use_va, "Trying to get kl_loss when not using va"
        return self.encoder.kl_loss() + self.processor.kl_loss() + self.decoder.kl_loss()
    
    def get_init_kwargs(self):
        return self.__init_kwargs__
