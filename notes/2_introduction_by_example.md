<h1> Pytorch Geometric学习系列 理论篇（二）--- 官方例子</h1>

Pytorch Geometric提供了官方例子来介绍其基础概念

<h2>1. 图的数据结构</h2>

Pytorch Geometric使用`torch_geometric.data.Data`数据结构来表示图，其默认包含以下参数/属性（这些参数/属性在构建实例的时候不是必须的，同时，也可以拓展参数/属性）：

+ `data.x`：节点特征矩阵，形状`[num_nodes, num_node_features]`；
+ `data.edge_index`：表示图邻接结构的COO格式稀疏矩阵，形状`[2, num_edges]`和数据类型`torch.long`；
+ `data.edge_attr`：边特征矩阵，形状`[num_edges, num_edge_features]`；
+ `data.y`：任务目标标签，可以是节点级别标签，也可以是图级别标签；
+ `data.pos`：节点位置矩阵，形状`[num_nodes, num_dimensions]`；

如下例子介绍如何使用`torch_geometric.data.Data`构建一个包含三个节点、两条边的无向无权图

```
import torch
from torch_geometric.data import Data
# 邻接矩阵
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
# 节点特征
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)
>>> Data(edge_index=[2, 4], x=[3, 1])
```

![](https://pytorch-geometric.readthedocs.io/en/latest/_images/graph.svg)

`edge_index`不是所有边索引的列表，形状`[num_edges, 2]`，而是其转置，形状`[2, num_edges]`：

```
import torch
from torch_geometric.data import Data

edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index.t().contiguous())
>>> Data(edge_index=[2, 4], x=[3, 1])
```

`torch_geometric.data.Data`提供了很多实用属性和方法：

```
print(data.keys)
>>> ['x', 'edge_index']

print(data['x'])
>>> tensor([[-1.0],
            [0.0],
            [1.0]])

for key, item in data:
    print("{} found in data".format(key))
>>> x found in data
>>> edge_index found in data

'edge_attr' in data
>>> False

data.num_nodes
>>> 3

data.num_edges
>>> 4

data.num_node_features
>>> 1

data.contains_isolated_nodes()
>>> False

data.contains_self_loops()
>>> False

data.is_directed()
>>> False

# Transfer data object to GPU.
device = torch.device('cuda')
data = data.to(device)
```

关于`torch_geometric.data.Data`的所有属性和方法，请参考[torch_geometric.data.Data](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data)

<h2>2. 常用图数据集</h2>

Pytorch Geometric内置提供大量的常用图数据集，所有planetoid数据集(Cora, Citeseer, Pubmed)，所有来自<http://graphkernels.cs.tu-dortmund.de/>的图分类数据集以及其[预处理版本](https://github.com/nd7141/graph_datasets)，QM7和QM9数据集，以及其他的3D mesh/point cloud数据集；

Pytorch Geometric使用`torch_geometric.data.Dataset`数据结构来表示数据集，本质上是`torch_geometric.data.Data`的派生类，实例化数据集主要完成的功能是下载数据集的原始文件，并预处理为`torch_geometric.data.Data`实例或实例列表；

更多关于Pytorch Geometric内置的图数据集的使用，请参考[官方教程](https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html)及[torch_geometric.data.Dataset](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Dataset)

<h3>3. Mini-batch</h3>

神经网络通常分批batch进行并行训练，图神经网络研究面临着多个稀疏图作为一个batch并行训练的困难，Pytorch Geomtric通过将多个稀疏邻接矩阵构建成一个大的对角块稀疏邻接矩阵(又`edge_index`和`edge_attr`定义)，以及拼接节点特征和标签，从而实现了多个稀疏图的并行训练，这样的拼接使得具有不同数目节点和数目边的图可以放进一个batch。

$$\begin{split}\mathbf{A} = \begin{bmatrix} \mathbf{A}_1 & & \\ & \ddots & \\ & & \mathbf{A}_n \end{bmatrix}, \qquad \mathbf{X} = \begin{bmatrix} \mathbf{X}_1 \\ \vdots \\ \mathbf{X}_n \end{bmatrix}, \qquad \mathbf{Y} = \begin{bmatrix} \mathbf{Y}_1 \\ \vdots \\ \mathbf{Y}_n \end{bmatrix}\end{split}$$


Pytorch Geometric提供`torch_geometric.data.DataLoader`来自动完成对图数据集中多个图的分批batch操作；下面例子通过使用`torch_geometric.data.Dataset`和`torch_geometric.data.DataLoader`来讲解图分批batch操作：

```
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    batch
    >>> Batch(batch=[1082], edge_index=[2, 4066], x=[1082, 21], y=[32])

    batch.num_graphs
    >>> 32
```

Pytorch Geometric使用`torch_geometric.data.Batch`来表示一批图数据，本质是`torch_geometric.data.Data`的派生类，包含一个额外的属性`batch`。

`batch`是一个列向量，用来映射batch中各个节点到各个图上；

$$\mathrm{batch} = {\begin{bmatrix} 0 & \cdots & 0 & 1 & \cdots & n - 2 & n -1 & \cdots & n - 1 \end{bmatrix}}^{\top}$$

我们可以通过使用`batch`属性来在各个图上对节点特征取平均；

```
from torch_scatter import scatter_mean
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for data in loader:
    data
    >>> Batch(batch=[1082], edge_index=[2, 4066], x=[1082, 21], y=[32])

    data.num_graphs
    >>> 32

    x = scatter_mean(data.x, data.batch, dim=0)
    x.size()
    >>> torch.Size([32, 21])
```

关于Pytorch Geometric的内部分批原理以及如何修改分批机制，请参考[这里](https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html)；对于scatter操作，请看[torch-scatter拓展库](https://pytorch-scatter.readthedocs.io/)

<h3>4. 数据变换</h3>

`torchvision`经常使用数据变换来变换图片和增强数据。Pytorch Geometric也定义了自己的数据变化`torch_geometric.transforms`，这些数据变换的输入为`Data`对象，返回变换后的`Data`对象。`torch_geometric.transforms`可以使用`torch_geometric.transforms.compose`来串接在一块，实现一系列的数据变换。

因为我的研究领域是图卷积神经网络，暂时没有使用到一些其他领域的数据，例如point cloud等等，所以略过，详情看官方文档。


<h3>5. 使用Pytorch Geometric实现图神经网络</h3>

在学习了Pytorch Geometric中的图数据结构、图数据集、数据装载器、数据变换，我们可以利用Pytorch Geometric来实现图神经网络。

下面例子介绍在Pytorch Geometric中使用一个简单的图卷积神经网络层来复现Cora Citation数据集上的实验。

加载Cora数据集

```
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='/tmp/Cora', name='Cora')
>>> Cora()
```

图卷积神经网络

```
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
```

训练过程

```
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
```

