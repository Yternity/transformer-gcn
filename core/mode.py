"""Model definitions for Transformer-GCN and baseline neural networks."""

# from layers import MLP,transformer,GNNLayer
# import torch.nn as nn
# import torch.nn.functional as F
# import torch
# class ourmodel(nn.Module):
#     def __init__(self, input_dim,hide,method,layers_count,A):
#         super(ourmodel, self).__init__()
#         self.A=A
#         self.method=method
#         self.lin = torch.nn.Linear(input_dim, hide)
#         self.bn = torch.nn.BatchNorm1d(hide)
#         self.trans = nn.Sequential()
#         for i in range(layers_count):
#             self.trans.add_module('trans' + str(i), transformer(hide, hide))
#         self.gnn =  nn.Sequential()
#         for i in range(layers_count):
#             self.gnn.add_module('gnn' + str(i), GNNLayer(hide, hide,self.A))
#         self.Softmax_linear = nn.Sequential(nn.Linear(hide, hide))
#     def forward(self, feature):
#         feature= F.normalize(feature)
#         feature = self.bn(self.lin(feature))
#         h1 =  self.trans(feature)+self.gnn(feature)
#         h2 =self.Softmax_linear(h1)

#         return F.log_softmax(h2, dim=1)

try:
    from .layers import MLP, transformer, GNNLayer
except ImportError:
    from layers import MLP, transformer, GNNLayer
import torch.nn as nn
import torch.nn.functional as F
import torch

# 提出的方法：Transformer + GCN
class ourmodel(nn.Module):
    """
    原始模型：先用线性层把输入特征压到隐藏维度，再堆叠若干层
    - transformer 模块（使用 MLP 模块实现）
    - GNNLayer（内部是 GCNConv）
    然后一个线性层得到最终分类 logits。
    """
    def __init__(self, input_dim, hide, method, layers_count, A):
        super(ourmodel, self).__init__()
        self.A = A
        self.method = method
        self.lin = torch.nn.Linear(input_dim, hide)
        self.bn = torch.nn.BatchNorm1d(hide)

        # Transformer-like 堆叠
        self.trans = nn.Sequential()
        for i in range(layers_count):
            self.trans.add_module(f"trans{i}", transformer(hide, hide))

        # 图卷积堆叠
        self.gnn = nn.Sequential()
        for i in range(layers_count):
            self.gnn.add_module(f"gnn{i}", GNNLayer(hide, hide, self.A))

        self.Softmax_linear = nn.Sequential(nn.Linear(hide, hide))

    def forward(self, feature):
        # 注意：train.py 里已经对 x 做过一次 normalize，这里再做一次问题不大
        feature = F.normalize(feature)
        feature = self.bn(self.lin(feature))
        h1 = self.trans(feature) + self.gnn(feature)
        h2 = self.Softmax_linear(h1)
        return F.log_softmax(h2, dim=1)


# ----------------------------------------------------------------------
# Baseline 1: 纯 MLP（不使用图结构）
# ----------------------------------------------------------------------
class MLPModel(nn.Module):
    """
    多层感知机：只用节点特征，不用图结构。
    用于对比“仅特征 + 深度网络”的性能。
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.normalize(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc_out(x)
        return F.log_softmax(x, dim=1)


# ----------------------------------------------------------------------
# Baseline 2: 经典 2-layer GCN
# ----------------------------------------------------------------------
from torch_geometric.nn import GCNConv, SAGEConv  # 用于 GCN / GraphSAGE 基线


class GCNModel(nn.Module):
    """
    经典 GCN baseline，在与主模型相同的空间 kNN 图上进行卷积。
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, edge_index):
        super().__init__()
        self.edge_index = edge_index
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, x):
        x = F.normalize(x)
        x = self.conv1(x, self.edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, self.edge_index)
        return F.log_softmax(x, dim=1)


# ----------------------------------------------------------------------
# Baseline 3: GraphSAGE
# ----------------------------------------------------------------------
class GraphSAGEModel(nn.Module):
    """
    GraphSAGE baseline，同样使用空间 kNN 图。
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, edge_index):
        super().__init__()
        self.edge_index = edge_index
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, num_classes)

    def forward(self, x):
        x = F.normalize(x)
        x = self.conv1(x, self.edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, self.edge_index)
        return F.log_softmax(x, dim=1)
