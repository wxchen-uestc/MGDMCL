import torch
import torch.nn as nn
import torch.nn.functional as F
from model.MaskGAE import AglMGae
from model.common_layer import LinearLayer, GateSelect, Graph, Linear, xavier_init
import utils


class Single_omics(nn.Module):
    def __init__(self, li_dim, num_class, k, dropout, mask_rate=0.5, a=1, b=1):
        super().__init__()
        rev_li_dim = li_dim.copy()
        rev_li_dim.reverse()
        self.layer = 'GCN'
        self.a, self.b = a, b
        self.mae = AglMGae(li_dim,
                           encoder_flag_graph=self.layer,
                           encoder_head_num=4,
                           encoder_flag_res=False,
                           encoder_flag_normalize=False,
                           encoder_activate_fun='leaky_relu',
                           k=k,
                           dropout=dropout,
                           mask_rate=mask_rate,
                           decoder_li_dim=rev_li_dim,
                           decoder_flag_graph=self.layer,
                           decoder_head_num=4,
                           decoder_flag_res=False,
                           decoder_flag_normalize=False,
                           decoder_activate_fun='leaky_relu')
        self.pre = LinearLayer(li_dim[-1], num_class)

    def forward(self, x, y=None, sample_weight=None, Training=False):
        feat_emb_li, loss_algmae, adj_gen = self.mae(x, y, Training)
        pre = self.pre(feat_emb_li[-1])
        # 计算损失函数
        loss = 0
        if Training:
            # 下游任务损失函数
            pre_criterion = nn.CrossEntropyLoss()
            loss_pre = torch.mean(torch.mul(pre_criterion(pre, y), sample_weight))
            # 挑选特征损失函数
            loss_att = loss_algmae['loss_att']
            # 特征重构函数
            loss_recon = loss_algmae['loss_recon']
            loss = (0 * loss_pre + self.a * loss_att + self.b * loss_recon)
        return feat_emb_li + [pre], loss, adj_gen


class Level_fusion(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, dropout):
        super().__init__()
        # setting parameters
        self.num_views = len(in_dim)
        self.dropout = dropout
        # 模型定义
        self.project = nn.ModuleList([LinearLayer(in_dim[_], hidden_dim) for _ in range(self.num_views)])
        self.LayerClassifierLayer = nn.ModuleList([LinearLayer(hidden_dim, num_class) for _ in range(self.num_views)])

    def forward(self, feature, label=None, sample_weight=None, Training=False):
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        LayerLogit = dict()
        all_cord = []
        # PDL
        feature_ = {}
        for layer in range(self.num_views):
            # project
            feature_[layer] = self.project[layer](feature[layer])
            feature_[layer] = F.relu(feature_[layer])
            # feature[layer] = F.dropout(feature[layer], self.dropout, training=self.training)
            # downstream tasks
            LayerLogit[layer] = self.LayerClassifierLayer[layer](feature_[layer])
            all_cord.append(feature_[layer])
        # concatenate
        MLfeature = torch.concatenate(all_cord, dim=1)
        # loss function
        Loss = 0
        if Training:
            # loss function for downstream tasks
            for layer in range(self.num_views):
                confidence_loss = torch.mean(torch.mul(criterion(LayerLogit[layer], label), sample_weight))
                Loss += 1 * confidence_loss
            # Multi-grained contrastive learning
            # fine-grained contrastive learning
            loss_fgcl_ab, loss_fgcl_ac, loss_fgcl_bc = 0, 0, 0
            loss_fgcl_ab = utils.sce_loss(feature_[0], feature_[1])
            loss_fgcl_ac = utils.sce_loss(feature_[0], feature_[2])
            loss_fgcl_bc = utils.sce_loss(feature_[1], feature_[2])
            loss_fgcl = loss_fgcl_ab + loss_fgcl_ac + loss_fgcl_bc
            # coarse-grained contrastive learning
            loss_cgcl = utils.SupervisedContrastiveLoss(MLfeature, label, 1)
            # sum
            Loss += 0.001 * loss_fgcl + 0.001 * loss_cgcl
        return MLfeature, Loss


class level_TCP(nn.Module):
    def __init__(self, in_dim, num_layers, num_class, dropout):
        super().__init__()
        # setting parameters
        self.num_layers = num_layers
        self.dropout = dropout
        # TCP
        self.TCPConfidenceLayer = nn.ModuleList([LinearLayer(in_dim[_], 1) for _ in range(self.num_layers)])
        self.TCPClassifierLayer = nn.ModuleList([LinearLayer(in_dim[_], num_class) for _ in range(self.num_layers)])
        #
        self.MMClasifier = LinearLayer(sum(in_dim), num_class)

    def forward(self, feature, y=None, sample_weight=None, Training=False):
        criterion = torch.nn.CrossEntropyLoss()
        TCPLogit, TCPConfidence = dict(), dict()
        all_cord = []
        # TCP
        for layer in range(self.num_layers):
            feature[layer] = F.relu(feature[layer])
            feature[layer] = F.dropout(feature[layer], self.dropout, training=self.training)
            TCPLogit[layer] = self.TCPClassifierLayer[layer](feature[layer])
            TCPConfidence[layer] = self.TCPConfidenceLayer[layer](feature[layer])
            TCPConfidence[layer] = F.sigmoid(TCPConfidence[layer])
            feature[layer] = feature[layer] * TCPConfidence[layer]
            all_cord.append(feature[layer])
        # concatenate
        # print(float(torch.mean(TCPConfidence[0]).detach().cpu()), float(torch.mean(TCPConfidence[1]).detach().cpu()),
        #       float(torch.mean(TCPConfidence[2]).detach().cpu()))
        MMfeature = torch.concatenate(all_cord, dim=1)
        MMlogit = self.MMClasifier(MMfeature)
        MMLoss = 0
        if Training:
            MMLoss = 1 * torch.mean(torch.mul(criterion(MMlogit, y), sample_weight))
            for layer in range(self.num_layers):
                pred = F.softmax(TCPLogit[layer], dim=1)
                p_target = torch.gather(input=pred, dim=1, index=y.unsqueeze(dim=1)).view(-1)
                confidence_loss = torch.mean(
                    F.mse_loss(TCPConfidence[layer].view(-1), p_target) + torch.mul(criterion(TCPLogit[layer], y), sample_weight))
                MMLoss += 0.1 * confidence_loss
        return MMfeature, MMlogit, MMLoss

