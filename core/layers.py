import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.nn import GATConv,SAGEConv,GCNConv


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.bn = nn.BatchNorm1d(output_dim)
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_normal_(self.weight, gain=1)
        if self.use_bias:
            init.normal_(self.bias, mean=0.0, std=0.01)

    def forward(self,input_feature):
        output=torch.mm(input_feature,self.weight)
        if self.use_bias:
            output+=self.bias
        output = self.bn(output)
        return F.relu(output)








class transformer(nn.Module):
    def __init__(self, input_dim,output_dim):
        super(transformer, self).__init__()
        self.mlp11 = MLP(input_dim, output_dim)
        self.mlp12 = MLP(input_dim, output_dim)
        self.mlp21 = MLP(output_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.mlp11.reset_parameters()
        self.mlp12.reset_parameters()
        self.mlp21.reset_parameters()

    def forward(self, x):
        Q = self.mlp11(x)
        V = self.mlp12(x)
        out = self.former(Q, V)
        ffn = (self.mlp21(out)+V)
        output = F.leaky_relu(self.bn(ffn))
        return output

    def former(self, Q, V):
        Q = F.layer_norm(Q, [Q.size(-1)])
        Q = F.relu(Q)
        out = torch.mm(Q, torch.mm(Q.T, V))
        norm = torch.mm(Q, (torch.sum(Q, dim=0)).unsqueeze(1))

        return F.relu(out / norm)








class GNNLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, A: torch.Tensor):#
        super(GNNLayer, self).__init__()
        self.A = A
        self.gcn = GCNConv(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)


    def forward(self, H):
        output = self.gcn(H,self.A)
        output = F.leaky_relu(self.bn(output))
        return output



