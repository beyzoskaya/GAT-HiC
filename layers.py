import torch
from torch import Tensor
from torch.nn import Parameter
from typing import Union, Tuple, Optional
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import GATConv
from torch_geometric.typing import OptPairTensor, Adj, Size, OptTensor
import torch.nn.functional as F

"""
This function is a version of customized GATConv if necessary for other implementations or different task model's to customized attention weights
"""

class CustomGATConv(GATConv):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, negative_slope=0.2, dropout=0, bias=True):
        super(CustomGATConv, self).__init__(in_channels, out_channels, heads=heads, concat=concat,
                                            negative_slope=negative_slope, dropout=dropout, bias=bias)
        
    def adjust_weights(self, adj_t: SparseTensor) -> SparseTensor:
        """Adjust weights based on neighborhood structure, returns SparseTensor."""
        sum_vec = adj_t.sum(dim=1).to(torch.float32)
        inverse = 1.0 / (sum_vec + 1e-12)  

        row = torch.arange(len(sum_vec), dtype=torch.long, device=adj_t.device())
        index = torch.stack([row, row], dim=0)
        value = inverse

        normalized = SparseTensor(row=index[0], col=index[1], value=value, sparse_sizes=(len(sum_vec), len(sum_vec)))
        norm_mat = matmul(normalized, adj_t)
        return norm_mat

    def forward(self, x, edge_index, return_attention_weights=False):
        x = (x, x) if isinstance(x, Tensor) else x
        out, attn_weights = super().forward(x, edge_index, return_attention_weights=True)

        adjusted_weights = self.adjust_weights(attn_weights)

        if return_attention_weights:
            return out, adjusted_weights
        else:
            return out

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        norm_mat = self.adjust_weights(adj_t)

        if isinstance(norm_mat, SparseTensor) and isinstance(x, Tensor):
            return matmul(norm_mat, x, reduce=self.aggr)
        else:
            raise ValueError("norm_mat must be SparseTensor and x must be a dense Tensor for matmul compatibility.")

    
