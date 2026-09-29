from turtle import forward
import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool


num_atom_type = 119
num_chirality_tag = 3

num_bond_type = 5
num_bond_direction = 3 

class GINEConv(MessagePassing):
    def __init__(self, emb_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, emb_dim)
        )
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # 1. 加入自环：在图神经网络（GNN）中，节点有时需要“看到”自己的特征，以加强表达能力。
        #    add_self_loops 会给每个节点加一条从自己指向自己的边（自环）。
        #    返回的是新的 edge_index（边索引），包含了所有原有边+每个节点的自环边。
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # 2. 为自环添加边特征：
        #    GINE需要每条边有属性（这里edge_attr原本 shape=[num_edges, 2]）。
        #    由于自环是新加的，所以要新建 x.size(0) 条自环边的属性，这里用全0初始化，shape为 [num_nodes, 2]。
        self_loop_attr = torch.zeros(x.size(0), 2)

        #    将自环边的第0维（代表bond type）设为4，通常原始的bond type是[0,1,2,3]，
        #    这里‘4’专门代表“自环”这种特殊的边类型，跟数据中真实的化学键类型区分开。
        self_loop_attr[:,0] = 4

        #    保证自环属性的数据类型和设备（device）与原始edge_attr一致。
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)

        # 3. 拼接自环边属性：
        #    把原始所有边的属性（N条边）和自环边的属性（N个节点）合并，得到新的edge_attr， shape=[num_edges+num_nodes, 2]
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        # 4. 边属性embdding：
        #    edge_attr的每一条边都用两个离散标签（类型/方向），分别映射成emb_dim维的向量，然后相加作为这条边的embedding。
        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        # 5. 正式消息传递
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)


    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)

class GINet(nn.Module):
    def __init__(self, num_layer=5, emb_dim=300, feat_dim=256, drop_ratio=0, pool='mean'):
        super().__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINEConv(emb_dim))

        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'max':
            self.pool = global_max_pool
        elif pool == 'add':
            self.pool = global_add_pool
        
        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)

        self.out_lin = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim), 
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, self.feat_dim//2)
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        h = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

        h = self.pool(h, data.batch)
        h = self.feat_lin(h)
        out = self.out_lin(h)
        
        return h, out
