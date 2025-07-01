import torch
import torch.nn as nn
import torch.nn.functional as F
from common_layer import LinearLayer, GateSelect
from MaskGAE import AglMGae
import utils
import math


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class view_fusion(nn.Module):
    def __init__(self, in_dim, hid_dim, num_class, k, dropout=0.5, num_views=3):
        super().__init__()
        self.num_views = num_views
        self.dropout = dropout
        self.k = k
        # 映射
        self.project = nn.ModuleList([LinearLayer(in_dim[_], hid_dim, activate_fun='relu') for _ in range(num_views)])
        # AglMGae
        # weight_attn, weight_graph, weight_recon, weight_pre
        self.MGae = AglMGae(encoder_li_dim=[hid_dim * num_views, hid_dim, hid_dim],
                            encoder_flag_graph='GCN',
                            encoder_head_num=4,
                            encoder_flag_res=False,
                            encoder_flag_normalize=True,
                            encoder_activate_fun='leak_relu',
                            k=k,
                            dropout=0.5,
                            decoder_flag=2,
                            mask_way=2,
                            mask_rate=0.6,
                            decoder_li_dim=[hid_dim, hid_dim, hid_dim * num_views],
                            decoder_flag_graph='GCN',
                            decoder_head_num=4,
                            decoder_flag_res=False,
                            decoder_flag_normalize=True,
                            decoder_activate_fun='leak_relu')
        self.pre = LinearLayer(hid_dim, num_class)

    def forward(self, view_feat, y=None, sample_weight=None, Training=None):
        view_project, view_relationship, update_view_fusion = {}, {}, {}
        # 映射拼接
        for i in range(self.num_views):
            view_project[i] = self.project[i](view_feat[i])
        view_feat_cont = torch.concatenate(list(view_project.values()), dim=1)

        # MASK GAE
        view_fusion_feat, loss_AglMGae = self.MGae(view_feat_cont, y, sample_weight, Training)
        view_fusion_feat = view_fusion_feat[-1]
        # pre
        pre = self.pre(view_fusion_feat)
        pre = F.softmax(pre, dim=1)
        loss = 0
        if Training:
            # 模态映射损失函数
            loss_project = (utils.sce_loss(view_project[0], view_project[1]) +
                            utils.sce_loss(view_project[0], view_project[2]) +
                            utils.sce_loss(view_project[1], view_project[2])) / 3
            # 下游任务损失函数
            pre_criterion = nn.CrossEntropyLoss()
            loss_pre = torch.mean(torch.mul(pre_criterion(pre, y), sample_weight))

            loss = 1 * (1 * loss_AglMGae['loss_att'] + 1 * loss_AglMGae['loss_graph'] + 1 * loss_AglMGae[
                'loss_recon']) + \
                   0.001 * loss_project + \
                   1 * loss_pre
        return pre, loss


class relation(nn.Module):
    def __init__(self, dim, num_class, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.fc = LinearLayer(dim * 2, dim)
        self.pre = LinearLayer(dim, num_class)

    def forward(self, x1, x2, y=None, sample_weight=None, Training=None):
        # view_fusion, fusion
        fusion = self.fc(torch.concatenate([x1, x2], dim=1))
        fusion = fusion + x1
        # fusion = self.fc(x1 + x2)
        fusion = F.relu(fusion)
        # fusion = torch.mul(fusion, x1)
        fusion = F.normalize(fusion, dim=1)
        # pre
        pre = self.pre(fusion)
        loss = 0
        if Training:
            pre_criterion = nn.CrossEntropyLoss()
            # 下游任务损失函数
            loss = torch.mean(torch.mul(pre_criterion(pre, y), sample_weight))
        return fusion, loss


class VCDN(nn.Module):
    def __init__(self, num_view, num_cls, hvcdn_dim):
        super().__init__()
        self.num_cls = num_cls
        self.model = nn.Sequential(
            nn.Linear(pow(num_cls, num_view), hvcdn_dim),
            nn.LeakyReLU(0.25),
            nn.Linear(hvcdn_dim, num_cls)
        )
        self.model.apply(xavier_init)

    def forward(self, in_list):
        num_view = len(in_list)
        for i in range(num_view):
            in_list[i] = torch.sigmoid(in_list[i])
        x = torch.reshape(torch.matmul(in_list[0].unsqueeze(-1), in_list[1].unsqueeze(1)),
                          (-1, pow(self.num_cls, 2), 1))
        for i in range(2, num_view):
            x = torch.reshape(torch.matmul(x, in_list[i].unsqueeze(1)), (-1, pow(self.num_cls, i + 1), 1))
        vcdn_feat = torch.reshape(x, (-1, pow(self.num_cls, num_view)))
        output = self.model(vcdn_feat)

        return output


class TCP(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, dropout):
        super().__init__()
        self.views = len(in_dim)
        self.classes = num_class
        self.dropout = dropout

        self.TCPConfidenceLayer = nn.ModuleList([LinearLayer(hidden_dim[0], 1) for _ in range(self.views)])
        self.TCPClassifierLayer = nn.ModuleList([LinearLayer(hidden_dim[0], num_class) for _ in range(self.views)])

        self.MMClasifier = []
        for layer in range(1, len(hidden_dim) - 1):
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[0], hidden_dim[layer]))
            self.MMClasifier.append(nn.ReLU())
            self.MMClasifier.append(nn.Dropout(p=dropout))
        if len(self.MMClasifier):
            self.MMClasifier.append(LinearLayer(hidden_dim[-1], num_class))
        else:
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[-1], num_class))
        self.MMClasifier = nn.Sequential(*self.MMClasifier)

    def forward(self, feature, y=None, Training=False):
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        TCPLogit, TCPConfidence = dict(), dict()
        all_cord = []
        for view in range(self.views):
            feature[view] = F.relu(feature[view])
            feature[view] = F.dropout(feature[view], self.dropout, training=self.training)
            TCPLogit[view] = self.TCPClassifierLayer[view](feature[view])
            TCPConfidence[view] = self.TCPConfidenceLayer[view](feature[view])
            feature[view] = feature[view] * TCPConfidence[view]
            all_cord.append(feature[view])
        MMfeature = torch.concatenate(all_cord, dim=1)
        MMlogit = self.MMClasifier(MMfeature)
        MMLoss = 0
        if Training:
            MMLoss = torch.mean(criterion(MMlogit, y))
            for view in range(self.views):
                pred = F.softmax(TCPLogit[view], dim=1)
                p_target = torch.gather(input=pred, dim=1, index=y.unsqueeze(dim=1)).view(-1)
                confidence_loss = torch.mean(
                    F.mse_loss(TCPConfidence[view].view(-1), p_target) + criterion(TCPLogit[view], y))
                MMLoss = MMLoss + confidence_loss
        return MMLoss, MMlogit