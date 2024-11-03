import pickle
import math
import dgl
import os
import torch
import numpy as np
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter
from torch_geometric.utils import softmax
from typing import Optional
import torch_geometric.nn as gnn
from torch_geometric.utils import degree
import scipy.sparse as sp
from torch.nn import init
from torch_geometric.nn import GATConv, GCNConv
import warnings


warnings.filterwarnings("ignore")
# Feature Path
Feature_Path = "./Feature/"
# Seed
SEED = 2020
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(SEED)

# model parameters
ADD_NODEFEATS = 'all'  # all/atom_feats/psepose_embedding/no
USE_EFEATS = True  # True/False
MAP_CUTOFF = 14
DIST_NORM = 15

# INPUT_DIM
if ADD_NODEFEATS == 'all':  # add atom features and psepose embedding
    INPUT_DIM = 54 + 7 + 1
elif ADD_NODEFEATS == 'atom_feats':  # only add atom features
    INPUT_DIM = 54 + 7
elif ADD_NODEFEATS == 'psepose_embedding':  # only add psepose embedding
    INPUT_DIM = 54 + 1
elif ADD_NODEFEATS == 'no':
    INPUT_DIM = 54
HIDDEN_DIM = 256  # hidden size of node features
LAYER = 6  # the number of MGU layers
DROPOUT = 0.1


LEARNING_RATE = 1E-3
WEIGHT_DECAY = 0
BATCH_SIZE = 1
NUM_CLASSES = 2  # [not bind, bind]
NUMBER_EPOCHS = 70

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def embedding(sequence_name):
    pssm_feature = np.load(Feature_Path + "pssm/" + sequence_name + '.npy')
    hmm_feature = np.load(Feature_Path + "hmm/" + sequence_name + '.npy')
    seq_embedding = np.concatenate([pssm_feature, hmm_feature], axis=1)
    return seq_embedding.astype(np.float32)


def get_dssp_features(sequence_name):
    dssp_feature = np.load(Feature_Path + "dssp/" + sequence_name + '.npy')
    return dssp_feature.astype(np.float32)


def get_res_atom_features(sequence_name):
    res_atom_feature = np.load(Feature_Path + "resAF/" + sequence_name + '.npy')
    return res_atom_feature.astype(np.float32)


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    result = r_mat_inv @ mx @ r_mat_inv
    return result


def cal_edges(sequence_name, radius=MAP_CUTOFF):  # to get the index of the edges
    dist_matrix = np.load(Feature_Path + "distance_map_SC/" + sequence_name + ".npy")
    mask = ((dist_matrix >= 0) * (dist_matrix <= radius))
    adjacency_matrix = mask.astype(int)
    radius_index_list = np.where(adjacency_matrix == 1)
    radius_index_list = [list(nodes) for nodes in radius_index_list]
    return radius_index_list


def load_graph(sequence_name):
    dismap = np.load(Feature_Path + "distance_map_SC/" + sequence_name + ".npy")
    mask = ((dismap >= 0) * (dismap <= MAP_CUTOFF))
    adjacency_matrix = mask.astype(int)
    norm_matrix = normalize(adjacency_matrix.astype(np.float32))
    return norm_matrix


def graph_collate(samples):
    sequence_name, sequence, label, node_features, G, adj_matrix, pos = map(list, zip(*samples))
    label = torch.Tensor(label)
    G_batch = dgl.batch(G)
    node_features = torch.cat(node_features)
    adj_matrix = torch.Tensor(adj_matrix)
    pos = torch.cat(pos)
    pos = torch.Tensor(pos)
    return sequence_name, sequence, label, node_features, G_batch, adj_matrix, pos


class ProDataset(Dataset):
    def __init__(self, dataframe, radius=MAP_CUTOFF, dist=DIST_NORM,
                 psepos_path='./Feature/psepos/Train335_psepos_SC.pkl'):
        self.names = dataframe['ID'].values
        self.sequences = dataframe['sequence'].values
        self.labels = dataframe['label'].values
        self.residue_psepos = pickle.load(open(psepos_path, 'rb'))
        self.radius = radius
        self.dist = dist

    def __getitem__(self, index):
        sequence_name = self.names[index]
        sequence = self.sequences[index]
        label = np.array(self.labels[index])
        nodes_num = len(sequence)
        pos = self.residue_psepos[sequence_name]
        pos = torch.from_numpy(pos).type(torch.FloatTensor)
        sequence_embedding = embedding(sequence_name)
        structural_features = get_dssp_features(sequence_name)
        node_features = np.concatenate([sequence_embedding, structural_features], axis=1)
        node_features = torch.from_numpy(node_features)
        if ADD_NODEFEATS == 'all' or ADD_NODEFEATS == 'atom_feats':
            res_atom_features = get_res_atom_features(sequence_name)
            res_atom_features = torch.from_numpy(res_atom_features)
            node_features = torch.cat([node_features, res_atom_features], dim=-1)
        if ADD_NODEFEATS == 'all' or ADD_NODEFEATS == 'psepose_embedding':
            node_features = torch.cat([node_features, torch.sqrt(
                torch.sum(pos * pos, dim=1)).unsqueeze(-1) / self.dist], dim=-1)

        radius_index_list = cal_edges(sequence_name, MAP_CUTOFF)
        edge_feat = self.cal_edge_attr(radius_index_list, pos)

        G = dgl.DGLGraph()
        G.add_nodes(nodes_num)
        edge_feat = np.transpose(edge_feat, (1, 2, 0))
        edge_feat = edge_feat.squeeze(1)

        self.add_edges_custom(G,
                              radius_index_list,
                              edge_feat
                              )
        adj_matrix = load_graph(sequence_name)
        node_features = node_features.detach().numpy()
        node_features = node_features[np.newaxis, :, :]
        node_features = torch.from_numpy(node_features).type(torch.FloatTensor)

        return sequence_name, sequence, label, node_features, G, adj_matrix, pos

    def __len__(self):
        return len(self.labels)

    def cal_edge_attr(self, index_list, pos):
        pdist = nn.PairwiseDistance(p=2, keepdim=True)
        cossim = nn.CosineSimilarity(dim=1)
        distance = (pdist(pos[index_list[0]], pos[index_list[1]]) / self.radius).detach().numpy()
        cos = ((cossim(pos[index_list[0]], pos[index_list[1]]).unsqueeze(-1) + 1) / 2).detach().numpy()
        radius_attr_list = np.array([distance, cos])
        return radius_attr_list
    def add_edges_custom(self, G, radius_index_list, edge_features):
        src, dst = radius_index_list[1], radius_index_list[0]
        if len(src) != len(dst):
            print('source and destination array should have been of the same length: src and dst:', len(src), len(dst))
            raise Exception
        G.add_edges(src, dst)
        G.edata['ex'] = torch.tensor(edge_features)
class model_MGU(nn.Module):
    def __init__(self, nlayers, nfeat, nhidden, nclass, dropout):
        super(model_MGU, self).__init__()
        self.layer1=2
        self.layer2=nlayers
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.pool1 = Pool(0.5, nhidden, dropout)
        self.pool2 = Pool(0.6, nhidden, dropout)
        self.pool3 = Pool(0.4, nhidden, dropout)
        self.unpool = Unpool(dropout)

        self.linear = nn.Linear(nhidden, nhidden)

        self.embedding_lap_pos_enc = nn.Linear(8, nhidden)
        self.cat_lin = nn.Linear(nhidden * 2, nhidden)
        self.GATConv = GATConv(nhidden, nhidden)
        self.relu = nn.ReLU()
        self.GConv = GCNConv(nhidden, nhidden)
        self.fc = nn.Linear(nhidden, nhidden)
        self.attn_norm = nn.LayerNorm(nhidden, eps=1e-6)
        self.MultiAttention = MuliHeadAttention(nhidden, dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.gcn_norm = nn.LayerNorm(nhidden, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(nhidden, eps=1e-6)
        self.ffn = FeedForwardNetwork(nhidden, nhidden * 3, dropout)
        self.ffn_dropout = nn.Dropout(dropout)

    def generate_mask(self, adjacency_matrix):
        return (adjacency_matrix == 0)

    def create_edge_index(self, adjacency_matrix):
        edge_index = torch.nonzero(adjacency_matrix).t().contiguous()
        return edge_index.to(adjacency_matrix.device)

    def rebuild_adj_matrix(self, edge_index, num_nodes):
        device = edge_index.device
        adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float, device=device)
        adj_matrix[edge_index[0], edge_index[1]] = 1
        return adj_matrix

    def laplacian_positional_encoding(self, adj_matrix, pos_enc_dim=8):

        adj_matrix = adj_matrix.to(torch.float32)

        # 计算度矩阵
        degrees = adj_matrix.sum(dim=1).cpu().numpy().flatten()

        # 计算度矩阵的逆平方根
        D_inv_sqrt = sp.diags(np.power(degrees, -0.5).flatten(), 0)

        # 将邻接矩阵转换为 CPU 上的 NumPy 数组
        adj_matrix_np = adj_matrix.cpu().numpy()

        # 构建归一化的拉普拉斯矩阵
        L = np.eye(adj_matrix.shape[0]) - D_inv_sqrt @ adj_matrix_np @ D_inv_sqrt

        # 计算拉普拉斯矩阵的特征值和特征向量
        EigVal, EigVec = np.linalg.eigh(L)

        # 对特征值和特征向量进行排序
        idx = EigVal.argsort()
        EigVal, EigVec = EigVal[idx], EigVec[:, idx]

        # 如果特征向量数量不足，进行零填充
        if EigVec.shape[1] < pos_enc_dim + 1:
            EigVec = np.pad(EigVec, ((0, 0), (0, pos_enc_dim + 1 - EigVec.shape[1])), mode='constant')

        # 选择特征向量作为位置编码
        pos_enc = EigVec[:, 1:pos_enc_dim + 1]

        # 将结果转换回 PyTorch 张量，并移动到原始设备
        pos_enc = torch.tensor(pos_enc, dtype=torch.float32, device=adj_matrix.device)
        return pos_enc

    def MainModule(self, h, adj,edge_index, laplacian):
        x = h
        y = self.attn_norm(x)
        y = self.relu(y)
        laplacian = self.embedding_lap_pos_enc(laplacian.float())
        y = y + laplacian
        y = self.MultiAttention(y, y, y, adj)
        y = self.attn_dropout(y)
        y = self.gcn_norm(y)
        y = self.relu(y)
        x = y
        y1 = self.GConv(y, edge_index)
        y2 = self.GATConv(y, edge_index)
        y = torch.cat((y1, y2), dim=1)
        y = self.cat_lin(y)
        y = y + x
        y = self.attn_dropout(y)
        y = self.ffn_norm(y)
        y = self.relu(y)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        y=y+h
        return y
    def forward(self, x,adj_matrix=None):

        h = self.act_fn(self.fcs[0](x))

        for j in range(self.layer1):
            adj_matrix_1, pooled_h_1, pooled_idx_1 = self.pool1(adj_matrix, h)
            adj_matrix_2, pooled_h_2, pooled_idx_2 = self.pool2(adj_matrix_1, pooled_h_1)
            adj_matrix_3, pooled_h_3, pooled_idx_3 = self.pool3(adj_matrix_2, pooled_h_2)

            edge_index = self.create_edge_index(adj_matrix_3)
            laplacian = self.laplacian_positional_encoding(adj_matrix_3)
            for i in range(self.layer2):
                pooled_h_3=self.MainModule(pooled_h_3,adj_matrix_3,edge_index,laplacian)
            unpool_h_2 = self.unpool(pooled_h_3, pooled_h_2, pooled_idx_3)
            h_2 = self.act_fn(self.linear(pooled_h_2 + unpool_h_2))



            edge_index = self.create_edge_index(adj_matrix_2)
            laplacian = self.laplacian_positional_encoding(adj_matrix_2)
            for i in range(self.layer2):
                h_2 = self.MainModule(h_2, adj_matrix_2, edge_index, laplacian)
            unpool_h_1 = self.unpool(h_2, pooled_h_1, pooled_idx_2)
            h_1 = self.act_fn(self.linear(unpool_h_1 + pooled_h_1))



            edge_index = self.create_edge_index(adj_matrix_1)
            laplacian = self.laplacian_positional_encoding(adj_matrix_1)
            for i in range(self.layer2):
                h_1 = self.MainModule(h_1, adj_matrix_1, edge_index, laplacian)
            unpool_h = self.unpool(h_1, h, pooled_idx_1)
            h = self.act_fn(self.linear(unpool_h + h))



            edge_index = self.create_edge_index(adj_matrix)
            laplacian = self.laplacian_positional_encoding(adj_matrix)
            for i in range(self.layer2):
                h = self.MainModule(h, adj_matrix, edge_index, laplacian)


        h = F.dropout(h, self.dropout, training=self.training).to(device)
        output = self.fcs[-1](h)
        return output
class MGUPPIS(nn.Module):
    def __init__(self, nlayers, nfeat, nhidden, nclass, dropout):
        super(MGUPPIS, self).__init__()
        self.model_MGU = model_MGU(nlayers=nlayers, nfeat=nfeat, nhidden=nhidden, nclass=nclass,dropout=dropout)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.6, patience=10,min_lr=1e-6)
    def forward(self, x, adj_matrix):
        x = x.float().to(device)
        x = x.view([x.shape[0] * x.shape[1], x.shape[2]])
        output = self.model_MGU(x=x,adj_matrix=adj_matrix)
        return output

class MuliHeadAttention(nn.Module):
    def __init__(self, hidden_size, dropout_rate, head_size=6):
        super(MuliHeadAttention, self).__init__()
        self.head_size = head_size
        self.attn_size = attn_size = hidden_size // head_size
        self.scale = attn_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * attn_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, head_size * attn_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, head_size * attn_size, bias=False)

        initialize_weight(self.linear_q)
        initialize_weight(self.linear_k)
        initialize_weight(self.linear_v)

        self.attn_dropout = nn.Dropout(dropout_rate)

        self.output_layer = nn.Linear(head_size * attn_size, hidden_size, bias=False)
        initialize_weight(self.output_layer)

    def forward(self, q, k, v, adj_matrix):

        if q.dim() == 2:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)

        batch_size, seq_length, _ = q.size()



        if adj_matrix.dim() == 2:
            adj_matrix = adj_matrix.unsqueeze(0)


        adj_matrix = adj_matrix.unsqueeze(1).expand(batch_size, self.head_size, seq_length, seq_length)


        q = self.linear_q(q).view(batch_size, seq_length, self.head_size, self.attn_size).transpose(1, 2)
        k = self.linear_k(k).view(batch_size, seq_length, self.head_size, self.attn_size).transpose(1, 2)
        v = self.linear_v(v).view(batch_size, seq_length, self.head_size, self.attn_size).transpose(1, 2)


        q = q * self.scale

        x = torch.matmul(q, k.transpose(-2, -1))
        x = torch.mul(adj_matrix, x)

        x = torch.softmax(x, dim=-1)

        x = self.attn_dropout(x)
        x = torch.matmul(x, v)
        # 重塑和输出
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.head_size * self.attn_size)
        output = self.output_layer(x)


        if q.size(0) == 1:
            output = output.squeeze(0)

        return output


def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, fliter_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(hidden_size, fliter_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(fliter_size, hidden_size)
        initialize_weight(self.layer1)
        initialize_weight(self.layer2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x


class Pool(nn.Module):
    def __init__(self, k, in_dim, p):
        super(Pool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, g, h):
        weights = self.proj(h).squeeze()
        scores = self.sigmoid(weights)
        return self.top_k_graph(scores, g, h, self.k)

    def top_k_graph(self, scores, g, h, k):
        num_nodes = g.shape[0]
        values, idx = torch.topk(scores, max(2, int(k * num_nodes)))
        sorted_idx, _ = torch.sort(idx)
        new_h = h[sorted_idx]
        new_g = g[sorted_idx][:, sorted_idx]
        return new_g, new_h, sorted_idx


class Unpool(nn.Module):
    def __init__(self, drop_p=0):
        super(Unpool, self).__init__()
        self.drop = nn.Dropout(p=drop_p) if drop_p > 0 else nn.Identity()

    def forward(self, h, pre_h, idx):
        new_h = torch.zeros_like(pre_h)
        new_h[idx] = h
        return self.drop(new_h)