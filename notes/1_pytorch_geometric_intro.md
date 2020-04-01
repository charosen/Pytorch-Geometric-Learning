<h1> Pytorch Geometric学习系列 理论篇（一）--- Pytorch Geometric介绍</h1>

<h2>1. 介绍 </h2>

Pytorch Geometric是Pytorch的几何深度学习拓展库。

在入门图神经网络的过程中，大家肯定经常遇到到如下现象问题：

    1. 图神经网络算法的实现思路大致为先在全连接图上计算当前节点对图上所有节点的注意力权重，再在聚合邻居特征时使用掩码mask以去除对非邻居节点特征的聚合，例如GCN, GAT等；
    2. 上述实现的核心是使用掩码mask来引入图的结构（节点`i`与哪些节点相连，与哪些节点又不相连）。尽管实现了图神经网络，但是全连接图上的计算占用了大量运算资源和GPU显存，并不是必要的，限制了为更复杂的图结构实现图神经网络算法，例如异质图，边有类型的图、边有特征的图；
    3. 现实中大多数图是稀疏的，同时，稀疏的图才是有价值的，才能更好的帮助图中节点聚合特征，在全连接图上聚合节点特征是无意义的。因而，图结构可以表示为稀疏矩阵，可以使用稀疏矩阵运算来引入图结构，并避免大量占用资源的全连接图运算；
    4. 当前知名的深度学习框架（Tensorflow、Pytorch等）对稀疏矩阵运算的支持十分差劲，例如仅支持加减乘除等简单运算、返回结果仍为占用大量显存的密集矩阵、不支持多个图的批并行处理等等；

Pytorch Geometric提供了方便的稀疏图数据结构、多图批处理、稀疏矩阵运算，因而，解决了上述的痛点问题，被广泛应用于当下热门的图神经网络的研究。

Pytorch Geometric主要提供了如下组件：

1. 稀疏图数据结构`torch_geometric.data`：提供了极具表示能力的图数据结构、诸多强大的图操作，以及最为重要的多图批处理；
2. 图数据集`torch_geometric.dataset`：提供了使用`torch_geometric.data`表示的、图神经网络研究中广泛使用的图数据集；
3. 实用功能`torch_geometric.utils`：提供了诸多图结构上的实用操作函数，包括稀疏图上的softmax操作，稀疏图上的dropout操作、为稀疏图添加自环、为稀疏图去除自环、计算图的拉普拉斯矩阵、构建子图、无向图转换等
4. 图神经网络`torch_geometric.nn`：内建实现许多图神经网络模型，以及基本的图卷积层、图卷积操作、图池化操作等；
5. 图转换`torch_geometric.transform`
6. 图输入输出操作`torch_geometric.io`


<h2>2. 安装</h2>

安装方法参照[Pytorch Geometric安装官方教程](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)，主要需要
1. 检查Pytorch的CUDA版本；
2. 正确安装配置CUDA和CuDNN；
3. 检查Pytorch的CUDA版本和系统CUDA版本；


<h2>3. 参考链接</h2>

1. <https://pytorch-geometric.readthedocs.io/en/latest/index.html>
2. <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html>
