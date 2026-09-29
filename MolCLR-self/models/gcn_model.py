import math

from sympy import zeros
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ReLU

from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils.num_nodes import maybe_num_nodes

num_atom_type = 119
num_chirality_tag = 3
num_bond_type = 5
num_bond_direction = 3


def gcn_norm(edge_index, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

    # edge_index.size(1) 返回edge_index的第二个维度的长度，即边的数量（edge数），
    # 假设edge_index的shape通常为 [2, num_edges]，那么size(1) 就是num_edges。

    # edge_index[1] 表示取edge_index这个2行num_edges列的矩阵的第2行（即索引为1的那一行），
    # 实际上是一个长度为num_edges的tensor，代表每条边的目标节点（每条边的终止点的编号）。

    # 换句话说：
    # - edge_index.size(1)：是一个整数，表示一共多少条边。
    # - edge_index[1]：是一个一维tensor，记录所有边的“目标节点”的索引。
    row, col = edge_index[0], edge_index[1]  # row 表示每条边的起点节点，col 表示每条边的终点节点

    # 下面几行代码的作用如下：
    from torch_scatter import scatter_add
    # 计算每个节点（col表示每条边的终止节点）的度数(degree)，即有多少条边指向该节点。
    # scatter_add的作用是，把edge_weight中属于同一个col索引（即同一个节点的所有边的权重1）累加起来，得到每个节点的入度（如果是无向图，可以理解为节点的度数）。

    # scatter_add的用法是：对于每个col中的节点索引，把edge_weight里对应位置的值加到对应节点的聚合结果上。
    # 由于edge_weight全是1，这里实际就是统计每个节点在col中出现的次数，也就是每个节点作为目标节点被多少条边指向——这就是“入度”。
    # 例如：如果某节点i在col中出现了3次，就说明有3条边指向i，scatter_add的结果deg[i]=3。
    # 不是很懂scatter_add的用法，需要进一步学习。
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    # 计算每个节点度数的-0.5次方（即 1/sqrt(deg)），后续用于GCN归一化（归一化邻接矩阵的一部分）。
    deg_inv_sqrt = deg.pow_(-0.5)
    # 避免某些节点度数为0导致无穷大，这里把无穷大的值用0填充，保证数值稳定性。
    deg_inv_sqrt.mask_fill_(deg_inv_sqrt == float('inf'), 0)

    # 这里要返回 deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]，
    # 是因为这正是GCN (Graph Convolutional Network) 论文中的归一化邻接矩阵的权重计算方式。
    # GCN中，每条边(i, j)的归一化权重为 1/sqrt(deg[i]) * 1 * 1/sqrt(deg[j])，
    # 其中deg[i]和deg[j]分别是边的起点和终点节点的度，因此用deg_inv_sqrt[row]和deg_inv_sqrt[col]分别表示，
    # edge_weight一般为1（如果有边权可以乘对应权重）。
    # 这样，消息在传播时每条边会按此因子缩放，实现正则化，避免度高节点影响过大，提升训练稳定性和泛化能力。
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class GCNConv(MessagePassing):  # 继承了MessagePassing类，MessagePassing类是PyG中用于实现消息传递的基类
    def __init__(self, emb_dim, aggr="add"):
        super().__init__()
        self.emb_dim = emb_dim
        self.aggr = aggr

        # 定义一个可学习的大小为 emb_dim * emb_dim 的权重矩阵
        self.weight = Parameter(torch.Tensor(emb_dim, emb_dim))
        # 定义一个可学习的大小为 emb_dim 的偏置向量
        self.bias = Parameter(torch.Tensor(emb_dim))
        self.reset_parameters()  # 调用reset_parameters方法对权重和偏置进行初始化

        self.edge_embedding1 = nn.Embedding(num_bond_type, 1)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, 1)

        # Xavier Uniform 初始化
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def reset_parameters(self):
        # glorot(self.weight)
        # zeros(self.bias)
        stdv = math.sqrt(6.0 / (self.weight.size(-2) + self.weight.size(-1)))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_attr):
        """
        x: 输入节点特征，形状为 (num_nodes, emb_dim)
        edge_index: 边索引，形状为 (2, num_edges)
        edge_attr: 边属性，形状为 (num_edges, 2)
        
        return: 输出特征，形状为 (num_nodes, emb_dim)
        """
        # 加入自环，新的节点表示 = 邻居信息＋自身信息
        # PyG 的 add_self_loops 函数 通常会返回类似 (edge_index, edge_attr) 的元组
        # [0] 表示取元组中的第一个元素，即 edge_index
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        self_loop_attr = torch.zeros(x.size(0), 2)

        self_loop_attr[:, 0] = 4

        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)

        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        # 把原始的离散边属性，转换成神经网络可以学习的连续 embedding 向量
        # edge_attr[:,0] 表示取 edge_attr 的第0列，即边类型（bond type）
        # edge_attr[:,1] 表示取 edge_attr 的第1列，即边方向（bond direction）
        # self.edge_embedding1 和 self.edge_embedding2 是两个可学习的嵌入层，用于将边类型和边方向转换成连续的 embedding 向量
        # 将两个嵌入向量相加，得到最终的边 embedding
        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        edge_index, __ = gcn_norm(edge_index)

        # x.shape = [N , emb_dim], weight.shape = [emb_dim , emb_dim]
        # 得到一个 [N , emb_dim] 的矩阵，即每个节点都经过同一个可学习矩阵weight进行特征变换
        x = x @ self.weight

        # 按照 edge_index 指定的图结构，让节点之间开始进行消息传递，并且把边的 embedding 一起传进去
        out = self.propagate(edge_index, x=x, edge_attr=edge_embeddings, size=None)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j, edge_attr):
        # return x_j if edge_attr is None else edge_attr.view(-1, 1) * x_j

        # x_i: 当前节点的特征
        # x_j: 邻居节点的特征
        # edge_attr: 边的特征
        # 如果边属性为空，则返回邻居节点的特征
        # 否则，将边属性与邻居节点特征相加，得到新的邻居节点特征
        return x_j if edge_attr is None else edge_attr + x_j

    def message_and_aggregate(self, adj_t, x):
        return torch_sparse.matmul(adj_t, x, reduce=self.aggr)


class GCN(nn.Module):
    def __init__(self, num_layers=5, emb_dim=300, feat_dim=256, drop_ratio=0, pool='mean'):
        super().__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.feat_dim = feat_dim
        self.emb_dim = emb_dim

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)

        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        self.gnns = nn.ModuleList()
        for layer in range(num_layers):
            self.gnns.append(GCNConv(emb_dim, aggr='add'))

        self.batch_norms = nn.ModuleList()
        for layer in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'add':
            self.pool = global_add_pool
        elif pool == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError('Not defined pooling!')

        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)

        self.out_lin = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, self.feat_dim // 2),
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        for layer in range(self.num_layers):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layers - 1:
                h = F.dropout(h, p=self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), p=self.drop_ratio, training=self.training)

        h = self.pool(h, data.batch)
        h = self.feat_lin(h)
        out = self.out_lin(h)
        return h, out


if __name__ == '__main__':
    model = GCN()
    print(model)
