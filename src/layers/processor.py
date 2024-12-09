from torch import nn
from torch import Tensor
from typing import Dict, Callable

from torch_geometric.typing import Metadata

from layers.interaction_network import InteractionNetworkBlock


class ProcessorModule(nn.Module):
    """Complete processor with multiple heterogeneous IN blocks"""

    def __init__(self, hidden_size: int, num_message_passing_steps: int, to_hetero: Callable, metadata: Metadata):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_message_passing_steps = num_message_passing_steps

        self.blocks = nn.ModuleList([InteractionNetworkBlock(hidden_size) for _ in range(num_message_passing_steps)])

        self.blocks = nn.ModuleList([to_hetero(block, metadata) for block in self.blocks])

        self.reset_parameters()

    def reset_parameters(self):
        for block in self.blocks:
            block.reset_parameters()

    def forward(self, x_dict: Dict[str, Tensor], edge_index_dict: Dict[str, Tensor], edge_attr_dict: Dict[str, Tensor]):
        current_x = x_dict
        current_edge_attr = edge_attr_dict

        for block in self.blocks:
            current_x, current_edge_attr = block(current_x, edge_index_dict, current_edge_attr)

        return current_x, current_edge_attr
