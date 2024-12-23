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

# information is aggregated from a node’s neighbors and combined with its own features

class SAGEConv(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 root_weight: bool = True,
                 bias: bool = True, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', 'add')
        super(SAGEConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        #print(f"Number of input channels: {in_channels}")
        self.out_channels = out_channels
        #print(f"Number of output channels: {out_channels}")
        self.normalize = normalize
        self.root_weight = root_weight
        
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False) # fully connected layer

        self.reset_parameters
         
    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
          self.lin_r.reset_parameters()

    def adjust_weights(self, adj_t):
      #device = torch.device('cuda')
      sum_vec = adj_t.sum(dim=0)
      numerator = torch.ones(len(sum_vec))
      inverse = torch.divide(numerator, sum_vec)
      size = len(sum_vec)

      row = torch.arange(size, dtype=torch.long)
      index = torch.stack([row, row], dim=0)

      value = inverse.float()
      normalized = SparseTensor(row=index[0], col=index[1], value=value)
      norm_mat = matmul(normalized, adj_t)
      return norm_mat


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr: OptTensor = None,
                size: Size = None) -> Tensor:
        if isinstance(x, Tensor):
          x: OptPairTensor = (x, x)
        
        out = self.propagate(edge_index, x=x, size=size)
        out = self.lin_l(out.float())
        x_r = x[1].long()
      
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r.float())

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)
        

        return out

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:

        norm_mat = self.adjust_weights(adj_t)
        return  matmul(norm_mat, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

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

    