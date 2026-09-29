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
    Chem.rdchem.ChiralType.CHI_OTHER,
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
    smiles_list = []
    with open(data_path, 'r') as f:
        reader = csv.reader(f)
        for _, row in enumerate(reader):
            smiles = row[-1]
            smiles_list.append(smiles)
    return smiles_list

class MoleculeDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.smiles_list = read_smiles(data_path)

    def __getitem__(self, idx):
        mol = Chem.MolFromSmiles(self.smiles_list[idx])

        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        atomic_number = []

        for atom in mol.GetAtoms():
            # 这里对每个原子提取其原子类型和手性，并映射为序号，便于后续embedding
            # ATOM_LIST 是包含所有可能原子序数的列表(1~118)，用于把原子的原子序（AtomicNum）映射成离散编号
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            # CHIRALITY_LIST 是所有可能手性标签的列表，用于将原子的手性标签映射为离散编号
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            # atomic_number 记录原子的原子序, 主要用于分析或后续可选的扩展特征
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        x = torch.cat([x1, x2], dim=-1)

        row, col, edge_feature = [], [], []

        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]

            edge_feature.append(
                [
                    BOND_LIST.index(bond.GetBondType()),
                    BONDDIR_LIST.index(bond.GetBondDir())
                ]
            )

            edge_feature.append(
                [
                    BOND_LIST.index(bond.GetBondType()),
                    BONDDIR_LIST.index(bond.GetBondDir())
                ]
            )
        
        edge_index = torch.tensor([row,col], dtype=torch.long) # 将 27 条边拆分成了 54条有向边 shape [2,54]
        edge_attr = torch.tensor(edge_feature, dtype=torch.long) # 给每条有向边都加入了化学键信息，包括化学键类型和化学键方向 [54,2]

        num_mask_nodes = max(1, math.floor(0.25 * N))
        num_mask_edges = max(0, math.floor(0.25 * M))

        # 从所有原子（节点）中随机选择 num_mask_nodes 个不同的下标，作为需要mask的节点
        mask_node_i = random.sample(range(N), num_mask_nodes)

        # 这里加不加 list 会有不同影响：
        # 在 Python 3 中，range(N) 本身是一个 range 对象，不是列表。
        # random.sample 要求传入的第一个参数是序列（如 list/tuple），它支持对 range 对象采样（Python 3.6+），
        # 但如果是 Python 2 或希望显式得到下标组成的列表，建议加 list — 否则得到的就是 range 对象（惰性序列）。
        # 若仅关注结果作为索引、且用的是 Python 3.6+，直接 random.sample(range(N), ...) 其实没区别。
        mask_node_j = random.sample(range(N), num_mask_nodes)

        mask_edges_i_single = random.sample(range(M), num_mask_edges)
        mask_edges_j_single = random.sample(range(M), num_mask_edges)

        # 解释：
        # 由于在上面edge的构建过程中，每条实际的化学键都被拆分成了2条有向边（正向和反向），
        # 给定单边下标，例如i，对应有向边的下标就是 2*i（正向）和 2*i+1（反向）。
        # 所以这里对于随机选中的要mask的单边索引 mask_edges_i_single，
        # 通过 [2*i for i in mask_edges_i_single] + [2*i+1 for i in mask_edges_i_single]
        # 就构造出了对应的有向边索引（正反向都mask）。
        mask_edges_i = [2*i for i in mask_edges_i_single] + [2*i+1 for i in mask_edges_i_single]
        mask_edges_j = [2*i for i in mask_edges_j_single] + [2*i+1 for i in mask_edges_j_single]

        x_i = deepcopy(x)

        # 这一句代码是在对被mask的节点在特征矩阵x_i中进行“mask处理”：将每个被选中mask的节点特征全部替换为特殊的mask token（如[len(ATOM_LIST), 0]），用于掩盖其原始原子类别及信息，通常用于自监督学习的掩码预测任务。
        for atom_idx in mask_node_i:
            # x_i[atom_idx,:] 表示对特征矩阵 x_i 的第 atom_idx 行（即第 atom_idx 个节点/原子），所有特征列（:）进行访问。
            # 这里将该节点的特征替换为 [len(ATOM_LIST), 0] 构成的张量，相当于把节点的原子类型和其他特征都mask为特定token。
            x_i[atom_idx,:] = torch.tensor([len(ATOM_LIST) + 1,0])
   
     
        # 这两句代码的作用：
        # edge_index_i：用于预先分配一个（2, 2*(M-num_mask_edges)）的张量，后面会装填去除了被mask边之后，剩余所有有向边的索引（每列一条边，行分别为起点和终点索引）；
        # edge_attr_i：同理，分配(2*(M-num_mask_edges), 2)张量，保存每条未被mask掉的有向边的属性特征（shape中每行为一条边的特征）。
        # 为什么这样预分配？后面会依次“搬运”未mask的边的edge信息进来，这样避免动态扩容，有利于效率和明确shape。
        edge_index_i = torch.zeros((2, 2 * (M - num_mask_edges)), dtype=torch.long)
        edge_attr_i = torch.zeros((2 * (M - num_mask_edges), 2), dtype=torch.long)

        # 这段代码的作用是根据选中的要mask的有向边，构建掩码后的新边索引和边属性矩阵（即去掉被mask的边以后剩下的边）。
        # 具体过程：
        # - 遍历所有有向边（bond_idx ∈ [0, 2*M)）。
        # - 对于每一个边，如果它没有被mask（即bond_idx不在mask_edges_i里）：
        #   - 就把原始的edge_index和edge_attr中该边的信息，按新的顺序依次填入edge_index_i和edge_attr_i（count记录新序号）。
        #   - count每填一条边加1。
        # 这样最终edge_index_i和edge_attr_i就是去除了被mask边之后的有向边信息。
        count = 0
        for bond_idx in range(2 * M):
            if bond_idx not in mask_edges_i:
                edge_index_i[:,count] = edge_index[:,bond_idx] # 将原始的edge_index中未被mask的边信息，按顺序填入edge_index_i中
                edge_attr_i[count,:] = edge_attr[bond_idx,:] # 将原始的edge_attr中未被mask的边信息，按顺序填入edge_attr_i中
                count += 1

        data_i = Data(x=x_i, edge_index=edge_index_i, edge_attr=edge_attr_i)

        x_j = deepcopy(x)
        for atom_idx in mask_node_j:
            x_j[atom_idx,:] = torch.tensor([len(ATOM_LIST)+1, 0])
        edge_index_j = torch.zeros((2, 2*(M-num_mask_edges)), dtype=torch.long)
        edge_attr_j = torch.zeros((2*(M-num_mask_edges), 2), dtype=torch.long)
        count = 0
        for bond_idx in range(2*M):
            if bond_idx not in mask_edges_j:
                edge_index_j[:,count] = edge_index[:,bond_idx]
                edge_attr_j[count,:] = edge_attr[bond_idx,:]
                count += 1
        data_j = Data(x=x_j, edge_index=edge_index_j, edge_attr=edge_attr_j)
        
        # data_i 和 data_j 分别是经过掩码(mask)处理后的两个 torch_geometric.data.Data 对象，作为一对正/负样本输入模型（通常用于图自监督或对比学习任务）。
        # 通常使用方法如下：
        #   - data_i 是第一种掩码方式下的子图（如对某些节点和边mask后），
        #   - data_j 是另一种掩码方式或mask补集下的子图，
        # 这两个子图可以输入到神经网络中分别提取图级（或节点级）表示，然后用于如对比损失(比如InfoNCE)中，作为一对正样本。
        # 这种设置常见于GraphCL、GraphMAE等自监督预训练方法，核心思想是让不同mask方式下图的表达保持一致（学习不变性）。
        return data_i, data_j

    def __len__(self):
        return len(self.smiles_list)


class MoleculeDatasetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_size, data_path):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size

    def get_train_validation_data_loaders(self, train_dataset):
        # 这三句代码的作用如下：
        # 1. 获取训练集(train_dataset)的样本数量，赋值给num_train。
        # 2. 创建一个从0到num_train-1的索引列表indices，表示所有样本的索引。
        # 3. 利用np.random.shuffle对indices进行原地随机打乱，实现训练集的随机顺序采样（常用于划分验证集或后续采样）。
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        # 这一步是划分验证集和训练集，即根据 valid_size 的比例确定验证集的样本数量 split
        split = int(np.floor(self.valid_size * num_train))
 
        train_idx, valid_idx = indices[split:], indices[:split]

        # 这两句代码是在根据已经划分好的训练集索引(train_idx)和验证集索引(valid_idx)，
        # 分别创建适用于 PyTorch DataLoader 的采样器(SubsetRandomSampler)。
        # 这样 DataLoader 就可以只从指定的子集（训练/验证子集）中按随机顺序采样样本，实现训练与验证集的分离与随机化。
        # 采样器（Sampler）在 PyTorch 的 DataLoader 数据加载机制中，决定了每个 epoch 里以什么顺序/方式选择数据。
        # SubsetRandomSampler 作用具体是：从一个给定的索引列表（比如 train_idx, valid_idx）中随机采样样本，无放回，每个 epoch 都会重新洗牌索引以保证训练的随机性。
        # 它用于将原始数据集划分后的子集（例如训练集或验证集）随机“洗牌”式采样，常用于对训练/验证过程的子集划分，以及保障样本分布的均匀性和增加泛化能力。
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True)
        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)
        return train_loader, valid_loader

    def get_data_loader(self):
        train_dataset = MoleculeDataset(self.data_path)
        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader


