<h1> Pytorch Geometric学习系列 理论篇（三）--- 构建图神经网络</h1>

<h2>1. 图神经网络的本质--消息传播/邻居聚合</h2>

非欧域的卷积/图卷积可以统一表示为一种邻居聚合函数/消息传播函数。给定$\mathbf{x}^{(k-1)}_i \in \mathbb{R}^F$表示k-1层卷积层的第i个节点的节点特征，以及$\mathbf{e}_{j,i} \in \mathbb{R}^D$表示节点j到节点i的边特征，图卷积神经网络可以表示为如下的消息传播函数：

$$\mathbf{x}_i^{(k)} = \gamma^{(k)} \left( \mathbf{x}_i^{(k-1)}, \square_{j \in \mathcal{N}(i)} \, \phi^{(k)}\left(\mathbf{x}_i^{(k-1)}, \mathbf{x}_j^{(k-1)},\mathbf{e}_{j,i}\right) \right)$$

其中，$\square$表示一种可微、具有置换不变性的函数，例如求和、取平均、取最大值；$\gamma$和$\phi$表示可微函数；

<h2>2. 使用Pytorch Geometric实现自己的图卷积 --- MessagePassing基类</h2>

Pytorch Geometric提供`torch_geometric.nn.MessagePassing`基类来帮助构建图卷积神经网络的消息传播函数，该基类自动处理了消息传播的整体流程（开始传播->计算消息/计算注意权重和消息加权->消息传播->消息更新）。用户只需要定义可微函数$\phi$，即消息函数`message()`（如何计算消息/注意力权重），以及可微函数$\gamma$，即更新函数，以及具体的聚合/传播机制，例如`aggr='add'`，`aggr='mean'`，`aggr='max'`（如何聚合/传播消息）；

`torch_geometric.nn.MessagePassing`基类提供了如下方法：
+ `torch_geometric.nn.MessagePassing(aggr="add", flow="source_to_target")`：定义了消息传播的聚合/传播可微函数，以及消息传播方向，如果是`"source_to_target"`则在行维度上聚合（行上求和），如果是`"target_to_source"`则在列维度上聚合（列上求和）；
+ `torch_geometric.nn.MessagePassing.message`：为节点i计算消息，当消息传播方向是`flow="source_to_target"`且存在(j,i)这条边时，或者，当消息传播方向是`flow="target_to_source"`且存在(i,j)这条边时。`propagate()`方法的所有参数都可以传递给`message()`，作为其参数。In addition, tensors passed to `propagate()` can be mapped to the respective nodes i and j by appending `_i` or `_j` to the variable name, .e.g. `x_i` and `x_j`.（这部分就是`__collect__`方法的功能）
+ `torch_geometric.nn.MessagePassing.update()`: Updates node embeddings in analogy to $\gamma$ for each node $i \in V$. Takes in the output of aggregation as first argument and any argument which was initially passed to `propagate()`.(消息聚合/传播之后的一些更新操作)

因此，使用Pytorch Geometric定义图卷积只需要
+ 继承`torch_geometric.nn.MessagePassing`基类
+ 自定义初始化方法（决定消息流向、聚合/传播方式）
+ 自定义前向传播方法`forward()`（前期张量准备工作，调用`propagate()`）
+ 自定义消息计算方法`message()`
+ 自定义消息更新方法`update()`。

要想理解整个消息传播流程，关键是看懂`torch_geometric.nn.MessagePassing`的初始化方法（为消息传播过程中的各个方法定义了很多特殊参数配置）以及`__collect__`方法（聚合所有参数，并分配给消息传播的各个方法，最重要的是定义了如何选取特定边的节点特征，为消息计算提供稀疏的节点特征）

完整代码：

```
import inspect
from collections import OrderedDict

import torch
from torch_geometric.utils import scatter_

msg_special_args = set([
    'edge_index',
    'edge_index_i',
    'edge_index_j',
    'size',
    'size_i',
    'size_j',
])

aggr_special_args = set([
    'index',
    'dim_size',
])

update_special_args = set([])


class MessagePassing(torch.nn.Module):
    r"""Base class for creating message passing layers

    .. math::
        \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i,
        \square_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}
        \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{i,j}\right) \right),

    where :math:`\square` denotes a differentiable, permutation invariant
    function, *e.g.*, sum, mean or max, and :math:`\gamma_{\mathbf{\Theta}}`
    and :math:`\phi_{\mathbf{\Theta}}` denote differentiable functions such as
    MLPs.
    See `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/
    create_gnn.html>`__ for the accompanying tutorial.

    Args:
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"` or :obj:`"max"`).
            (default: :obj:`"add"`)
        flow (string, optional): The flow direction of message passing
            (:obj:`"source_to_target"` or :obj:`"target_to_source"`).
            (default: :obj:`"source_to_target"`)
        node_dim (int, optional): The axis along which to propagate.
            (default: :obj:`0`)
    """
    def __init__(self, aggr='add', flow='source_to_target', node_dim=0):
        super(MessagePassing, self).__init__()

        self.aggr = aggr
        assert self.aggr in ['add', 'mean', 'max']

        self.flow = flow
        assert self.flow in ['source_to_target', 'target_to_source']

        self.node_dim = node_dim
        assert self.node_dim >= 0

        self.__msg_params__ = inspect.signature(self.message).parameters
        self.__msg_params__ = OrderedDict(self.__msg_params__)

        self.__aggr_params__ = inspect.signature(self.aggregate).parameters
        self.__aggr_params__ = OrderedDict(self.__aggr_params__)
        self.__aggr_params__.popitem(last=False)

        self.__update_params__ = inspect.signature(self.update).parameters
        self.__update_params__ = OrderedDict(self.__update_params__)
        self.__update_params__.popitem(last=False)

        msg_args = set(self.__msg_params__.keys()) - msg_special_args
        aggr_args = set(self.__aggr_params__.keys()) - aggr_special_args
        update_args = set(self.__update_params__.keys()) - update_special_args

        self.__args__ = set().union(msg_args, aggr_args, update_args)

    def __set_size__(self, size, index, tensor):
        if not torch.is_tensor(tensor):
            pass
        elif size[index] is None:
            size[index] = tensor.size(self.node_dim)
        elif size[index] != tensor.size(self.node_dim):
            raise ValueError(
                (f'Encountered node tensor with size '
                 f'{tensor.size(self.node_dim)} in dimension {self.node_dim}, '
                 f'but expected size {size[index]}.'))

    def __collect__(self, edge_index, size, kwargs):
        i, j = (0, 1) if self.flow == "target_to_source" else (1, 0)
        ij = {"_i": i, "_j": j}

        out = {}
        for arg in self.__args__:
            if arg[-2:] not in ij.keys():
                out[arg] = kwargs.get(arg, inspect.Parameter.empty)
            else:
                idx = ij[arg[-2:]]
                data = kwargs.get(arg[:-2], inspect.Parameter.empty)

                if data is inspect.Parameter.empty:
                    out[arg] = data
                    continue

                if isinstance(data, tuple) or isinstance(data, list):
                    assert len(data) == 2
                    self.__set_size__(size, 1 - idx, data[1 - idx])
                    data = data[idx]

                if not torch.is_tensor(data):
                    out[arg] = data
                    continue

                self.__set_size__(size, idx, data)
                out[arg] = data.index_select(self.node_dim, edge_index[idx])

        size[0] = size[1] if size[0] is None else size[0]
        size[1] = size[0] if size[1] is None else size[1]

        # Add special message arguments.
        out['edge_index'] = edge_index
        out['edge_index_i'] = edge_index[i]
        out['edge_index_j'] = edge_index[j]
        out['size'] = size
        out['size_i'] = size[i]
        out['size_j'] = size[j]

        # Add special aggregate arguments.
        out['index'] = out['edge_index_i']
        out['dim_size'] = out['size_i']

        return out

    def __distribute__(self, params, kwargs):
        out = {}
        for key, param in params.items():
            data = kwargs[key]
            if data is inspect.Parameter.empty:
                if param.default is inspect.Parameter.empty:
                    raise TypeError(f'Required parameter {key} is empty.')
                data = param.default
            out[key] = data
        return out

    def propagate(self, edge_index, size=None, **kwargs):
        r"""The initial call to start propagating messages.

        Args:
            edge_index (Tensor): The indices of a general (sparse) assignment
                matrix with shape :obj:`[N, M]` (can be directed or
                undirected).
            size (list or tuple, optional): The size :obj:`[N, M]` of the
                assignment matrix. If set to :obj:`None`, the size will be
                automatically inferred and assumed to be quadratic.
                (default: :obj:`None`)
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """

        size = [None, None] if size is None else size
        size = [size, size] if isinstance(size, int) else size
        size = size.tolist() if torch.is_tensor(size) else size
        size = list(size) if isinstance(size, tuple) else size
        assert isinstance(size, list)
        assert len(size) == 2

        kwargs = self.__collect__(edge_index, size, kwargs)

        msg_kwargs = self.__distribute__(self.__msg_params__, kwargs)
        out = self.message(**msg_kwargs)

        aggr_kwargs = self.__distribute__(self.__aggr_params__, kwargs)
        out = self.aggregate(out, **aggr_kwargs)

        update_kwargs = self.__distribute__(self.__update_params__, kwargs)
        out = self.update(out, **update_kwargs)

        return out

    def message(self, x_j):  # pragma: no cover
        r"""Constructs messages to node :math:`i` in analogy to
        :math:`\phi_{\mathbf{\Theta}}` for each edge in
        :math:`(j,i) \in \mathcal{E}` if :obj:`flow="source_to_target"` and
        :math:`(i,j) \in \mathcal{E}` if :obj:`flow="target_to_source"`.
        Can take any argument which was initially passed to :meth:`propagate`.
        In addition, tensors passed to :meth:`propagate` can be mapped to the
        respective nodes :math:`i` and :math:`j` by appending :obj:`_i` or
        :obj:`_j` to the variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`.
        """

        return x_j

    def aggregate(self, inputs, index, dim_size):  # pragma: no cover
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        By default, delegates call to scatter functions that support
        "add", "mean" and "max" operations specified in :meth:`__init__` by
        the :obj:`aggr` argument.
        """

        return scatter_(self.aggr, inputs, index, self.node_dim, dim_size)

    def update(self, inputs):  # pragma: no cover
        r"""Updates node embeddings in analogy to
        :math:`\gamma_{\mathbf{\Theta}}` for each node
        :math:`i \in \mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`propagate`.
        """

        return inputs

```

关于图卷积层的实现例子以及边卷积层的实现例子，请参考官网教程[Creating Message Passing Networks](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html)

Pytorch Geometric内置实现了很多图神经网络，并提供了大量例子，至于`examples`目录中，大家可以看着例子学习实现自己的图卷积操作。

<h2>2. 参考链接</h2>

1. [torch_geometric.nn](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html)
2. [Creating Message Passing Networks](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html)
