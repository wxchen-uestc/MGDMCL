import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def get_activate_fun(activate_fun):
    if activate_fun == 'relu':
        activate_fun = nn.ReLU()
    elif activate_fun == 'tanh':
        activate_fun = nn.Tanh()
    elif activate_fun == 'sigmoid':
        activate_fun = nn.Sigmoid()
    elif activate_fun == 'leaky_relu':
        activate_fun = nn.LeakyReLU(0.25)
    else:
        activate_fun = False
    return activate_fun


def get_graph_layer(layer_name, in_dim, out_dim, head_num=2):
    if layer_name == 'GCN':
        return GCNConv(in_dim, out_dim)
    elif layer_name == 'GAT':
        return GATConv(in_dim, out_dim, heads=head_num, concat=False)


def get_res_fc(in_dim, out_dim):
    return LinearLayer(in_dim, out_dim, activate_fun='leak_relu')


class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activate_fun=None, Is_normal=None):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.activate_fun = get_activate_fun(activate_fun)
        self.Is_normal = Is_normal
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        if self.activate_fun:
            x = self.activate_fun(x)
        if self.Is_normal:
            x = F.normalize(x, dim=1)
        return x


class Graph(nn.Module):
    def __init__(self, li_dim, flag_graph='GCN', head_num=4,
                 flag_res=False, flag_normalize=True, activate_fun=None, dropout=0.5):
        super().__init__()
        self.layer_num = len(li_dim) - 1
        self.graph_layer = []
        for i in range(self.layer_num):
            self.graph_layer.append(get_graph_layer(flag_graph, li_dim[i], li_dim[i + 1], head_num))
        if self.layer_num > 1:
            self.graph_layer.append(get_graph_layer(flag_graph, li_dim[-2], li_dim[-1], head_num))
        self.graph_layer = nn.ModuleList(self.graph_layer)
        self.flag_graph = flag_graph
        self.flag_res = flag_res
        self.flag_normalize = flag_normalize
        self.activate_fun = get_activate_fun(activate_fun)
        self.dropout = dropout

        if self.flag_res:
            self.res_fc = []
            for i in range(self.layer_num):
                self.res_fc.append(get_res_fc(li_dim[i], li_dim[i + 1]))
            if self.layer_num > 1:
                self.res_fc.append(get_res_fc(li_dim[-2], li_dim[-1]))
            self.res_fc = nn.ModuleList(self.res_fc)

    def forward(self, x, edge_index, edge_weight=None):
        feat = {-1: x}
        for layer in range(self.layer_num):
            feat[layer] = F.dropout(feat[layer - 1], self.dropout, training=self.training)
            if self.flag_graph == 'GCN':
                feat[layer] = self.graph_layer[layer](feat[layer], edge_index, edge_weight=edge_weight)
            elif self.flag_graph == 'GAT':
                feat[layer] = self.graph_layer[layer](feat[layer], edge_index, edge_attr=edge_weight)
            if self.activate_fun:
                feat[layer] = self.activate_fun(feat[layer])
            if self.flag_normalize:
                feat[layer] = F.normalize(feat[layer], dim=1)
            if self.flag_res:
                feat[layer] += self.res_fc[layer](feat[layer - 1])
        feat = list(feat.values())
        return feat


class Linear(nn.Module):
    def __init__(self, li_dim, flag_res=False, flag_normalize=True, activate_fun=None, dropout=0.5):
        super().__init__()
        self.layer_num = len(li_dim) - 1
        self.Linear_layer = []
        for i in range(self.layer_num):
            self.Linear_layer.append(LinearLayer(li_dim[i], li_dim[i + 1]))
        if self.layer_num > 1:
            self.Linear_layer.append(LinearLayer(li_dim[-2], li_dim[-1]))
        self.Linear_layer = nn.ModuleList(self.Linear_layer)
        self.flag_res = flag_res
        self.flag_normalize = flag_normalize
        self.activate_fun = get_activate_fun(activate_fun)
        self.dropout = dropout

        if self.flag_res:
            self.res_fc = []
            for i in range(self.layer_num):
                self.res_fc.append(get_res_fc(li_dim[i], li_dim[i + 1]))
            if self.layer_num > 1:
                self.res_fc.append(get_res_fc(li_dim[-2], li_dim[-1]))
            self.res_fc = nn.ModuleList(self.res_fc)

    def forward(self, x):
        feat = {-1: x}
        for layer in range(self.layer_num):
            feat[layer] = F.dropout(feat[layer - 1], self.dropout, training=self.training)
            feat[layer] = self.Linear_layer[layer](feat[layer])
            if self.activate_fun:
                feat[layer] = self.activate_fun(feat[layer])
            if self.flag_normalize:
                feat[layer] = F.normalize(feat[layer], dim=1)
            if self.flag_res:
                feat[layer] += self.res_fc[layer](feat[layer - 1])
        feat = list(feat.values())
        return feat


class GateSelect(nn.Module):
    def __init__(self, in_dim, dropout):
        super().__init__()
        self.att = LinearLayer(in_dim, in_dim)
        self.dropout = dropout

    def forward(self, x):
        att_score = torch.sigmoid(self.att(x))
        feat_emb = torch.mul(att_score, x)
        feat_emb = F.relu(feat_emb)
        feat_emb = F.dropout(feat_emb, self.dropout, training=self.training)

        return att_score, feat_emb

