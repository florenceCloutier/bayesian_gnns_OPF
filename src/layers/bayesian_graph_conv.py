import torch
from torch import Tensor
from typing import Final, Tuple, Union

from torch_geometric import EdgeIndex
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.utils import spmm

from .bayesian_linear import BayesianLinear


class BayesianGraphConv(MessagePassing):
    SUPPORTS_FUSED_EDGE_INDEX: Final[bool] = True

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: str = "add",
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        # Replace standard Linear layers with BayesianLinear
        self.lin_rel = BayesianLinear(in_channels[0], out_channels, bias=bias)
        self.lin_root = BayesianLinear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_rel.reset_parameters()
        self.lin_root.reset_parameters()

    def forward(
        self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_weight: OptTensor = None, size: Size = None
    ) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)

        # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)
        out = self.lin_rel(out)

        x_r = x[1]
        if x_r is not None:
            out = out + self.lin_root(x_r)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(
        self,
        edge_index: Adj,
        x: OptPairTensor,
        edge_weight: OptTensor,
    ) -> Tensor:
        if not torch.jit.is_scripting() and isinstance(edge_index, EdgeIndex):
            return edge_index.matmul(
                other=x[0],
                input_value=edge_weight,
                reduce=self.aggr,
                transpose=True,
            )
        return spmm(edge_index, x[0], reduce=self.aggr)

    def kl_loss(self):
        """Calculate total KL divergence for the layer"""
        return self.lin_rel.kl_loss() + self.lin_root.kl_loss()
