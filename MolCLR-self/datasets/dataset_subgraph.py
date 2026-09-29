import os
import csv
import math
import time
import random
import networkx as nx
import numpy as np
from copy import deepcopy

import torch
import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms

from torch_scatter import scatter
from torch_geometric.data import Data, Dataset, DataLoader

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem


ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [
    BT.SINGLE, 
    BT.DOUBLE, 
    BT.TRIPLE, 
    BT.AROMATIC
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]


def read_smiles(data_path):
    smiles_data = []
    with open(data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            smiles = row[-1]
            smiles_data.append(smiles)
            # mol = Chem.MolFromSmiles(smiles)
            # if mol != None:
            #     smiles_data.append(smiles)
    return smiles_data

def removeSubgraph(Graph, center, percent=0.2):
    """
    从原始分子图中移除一个连通子图。
    参数：
        1. 原始图（networkx 图对象）
        2. 移除操作起始原子的索引（整数）
        3. 需要从原始图中移除的原子数量占比

        输出：
        1. 移除子图后得到的结果图（networkx 图对象）
        2. 被移除原子的索引列表（列表）
    """
    assert percent >= 0 and percent <= 1
    G = Graph.copy()
    # 这句代码的意思是：先计算当前图 G 中节点（原子）的总数 len(G.nodes)，再乘以设定的百分比 percent，得到需要被移除的原子数量（可能是小数）。
    # 用 np.floor 向下取整得到一个不超过目标比例的整数，然后再用 int 转为整数类型。
    # 也就是说，这句代码会根据给定比例，计算出应该移除的原子（节点）数 num。
    num = int(np.floor(len(G.nodes) * percent))

    removed = []

    # temp 列表用于记录当前正在扩展的原子索引（从center开始），后续会以广度优先遍历的方式逐步找到与center连接的原子，
    # 直到达到需要移除的原子数量。这可以保证被移除的子图是连通的。
    temp = [center]
    while len(removed) < num:
        neighbors = []
        for n in temp:
            # 遍历当前要扩展的原子 n，用 G.neighbors(n) 得到其直接相连的邻居原子索引；
            # 对每个邻居原子索引 i，如果它还没有被加入到 temp（即不是已扩展队列）中，则收集进 neighbors 列表（用于下一轮拓展）。
            neighbors.extend([i for i in G.neighbors(n) if i not in temp])
        for n in temp:
            if len(removed) < num:
                G.remove_node(n)
                removed.append(n)
            else:
                break
        temp = list(set(neighbors))
    return G, removed # 返回移除子图后得到的结果图（networkx 图对象）和被移除原子的索引列表（列表）

class MoleculeDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.smiles_data = read_smiles(data_path)

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.smiles_data[index])

        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        atomic_number = []
        atoms = mol.GetAtoms()
        bonds = mol.GetBonds()

        # 这句代码的作用是：从当前分子的所有原子下标（0 到 N-1）中随机抽取两个互不相同的原子下标，分别赋值给 start_i 和 start_j。
        # random.sample(range(N), 2) 会在不放回的情况下随机返回 2 个不同的原子序号，用于后续扩展/裁剪子图等任务的起点选择。
        start_i, start_j = random.sample(range(N), 2)

        edges = []

        for bond in bonds:
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        # 用 edges 列表创建一个无向图（networkx 图对象），edges 中的每个元素是两个原子索引构成的边。
        molGraph = nx.Graph(edges)


        percent_i, percent_j = 0.25, 0.25
        G_i, removed_i = removeSubgraph(molGraph, start_i, percent_i)
        G_j, removed_j = removeSubgraph(molGraph, start_j, percent_j)

        for atom in atoms:
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        x = torch.cat([x1, x2], dim=-1)

        # Mask the atoms in the removed list
        x_i = deepcopy(x)
        for atom_idx in removed_i:
            # Change atom type to 118, and chirality to 0
            x_i[atom_idx,:] = torch.tensor([len(ATOM_LIST), 0])
        x_j = deepcopy(x)
        for atom_idx in removed_j:
            # Change atom type to 118, and chirality to 0
            x_j[atom_idx,:] = torch.tensor([len(ATOM_LIST), 0])

        # 这段代码的作用是为两个子图 G_i 和 G_j 分别构建边的索引和边特征
        # row_i、col_i 用于表示子图 G_i 中所有边的起点和终点索引
        # row_j、col_j 用于表示子图 G_j 中所有边的起点和终点索引
        # edge_feat_i、edge_feat_j 存储每条边的特征（如类型、方向等），一一对应于各自的边
        # G_i_edges 和 G_j_edges 分别为子图 G_i 和 G_j 的所有边（每条边用两个原子索引组成的元组表示，来自 networkx.Graph.edges）
        row_i, col_i, row_j, col_j = [], [], [], []
        edge_feat_i, edge_feat_j = [], []
        G_i_edges = list(G_i.edges)
        G_j_edges = list(G_j.edges)

        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            feature = [BOND_LIST.index(bond.GetBondType()),BONDDIR_LIST.index(bond.GetBondDir())]

            if (start, end) in G_i_edges:
                row_i += [start, end]
                col_i += [end, start]
                edge_feat_i.append(feature)
                edge_feat_i.append(feature)

            if (start, end) in G_j_edges:
                row_j += [start, end]
                col_j += [end, start]
                edge_feat_j.append(feature)
                edge_feat_j.append(feature)
        
        edge_index_i = torch.tensor([row_i, col_i], dtype=torch.long)
        edge_attr_i = torch.tensor(np.array(edge_feat_i), dtype=torch.long)
        edge_index_j = torch.tensor([row_j, col_j], dtype=torch.long)
        edge_attr_j = torch.tensor(np.array(edge_feat_j), dtype=torch.long)

        data_i = Data(x=x_i, edge_index=edge_index_i, edge_attr=edge_attr_i)
        data_j = Data(x=x_j, edge_index=edge_index_j, edge_attr=edge_attr_j)
        return data_i, data_j
    
    def __len__(self):
        return len(self.smiles_data)


class MoleculeDatasetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_size, data_path):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size

    def get_data_loaders(self):
        train_dataset = MoleculeDataset(data_path=self.data_path)
        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        
        # random_state = np.random.RandomState(seed=666)
        # random_state.shuffle(indices)
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True)

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)

        return train_loader, valid_loader

