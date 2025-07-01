import torch.nn as nn
from model.common_layer import LinearLayer
from model.single_omics import Single_omics, Level_fusion, level_TCP


class FUSION(nn.Module):
    def __init__(self, in_dim, li_hid_dim, k, num_class, dropout, mask_rate):
        super().__init__()
        # setting parameters
        self.num_views = len(in_dim)  # number of views
        self.num_layers = len(li_hid_dim)  # number of layers
        self.dropout = dropout
        # omics-specific learning i.e. dynamic graph reconstruction (feature selection + masked graph autoencoder)
        self.dict_single_omics = []
        for i in range(self.num_views):
            self.dict_single_omics.append(
                Single_omics([in_dim[i]] + li_hid_dim, num_class, k, dropout, mask_rate, a=0.01, b=0.01))
        self.dict_single_omics = nn.ModuleList(self.dict_single_omics)
        # Features Contrastive Learning
        self.dict_level_fusion = []
        for i in range(self.num_layers):
            self.dict_level_fusion.append(
                Level_fusion([li_hid_dim[i]] * self.num_views, li_hid_dim[i], num_class, dropout))
        self.dict_level_fusion = nn.ModuleList(self.dict_level_fusion)
        # Confidence Learning
        self.fusion = level_TCP([i * self.num_views for i in li_hid_dim], self.num_layers, num_class, dropout)
        # downstream tasks
        in_dim = sum(li_hid_dim)
        self.pre = LinearLayer(in_dim * self.num_views, num_class)

    def forward(self, x, y=None, sample_weight=None, matrix_same=None, Training=None):
        # omics-specific learning
        dict_view_loss, dict_view_feat, dict_view_graph = {}, {}, {}
        for i in range(self.num_views):
            dict_view_feat[i], dict_view_loss[i], dict_view_graph[i] = self.dict_single_omics[i](x[i], y,
                                                                                                 sample_weight,
                                                                                                 Training)
        # Features Contrastive Learning
        dict_level_fusion_feat, dict_level_fusion_loss = {}, {}
        for i in range(self.num_layers):
            level_fusion_feat = []
            for j in range(self.num_views):
                level_fusion_feat.append(dict_view_feat[j][i + 1])
            dict_level_fusion_feat[i], dict_level_fusion_loss[i] = \
                self.dict_level_fusion[i](level_fusion_feat, y, sample_weight, Training)
        # Confidence Learning
        MMfeature, pre, loss_pre = self.fusion(list(dict_level_fusion_feat.values()), y, sample_weight, Training)
        # print(1)
        ci_loss = 0
        if Training:
            pre_criterion = nn.CrossEntropyLoss()
            # loss_pre = torch.mean(torch.mul(pre_criterion(pre, y), sample_weight))
            # print(loss_pre)
            # print(1)
            ci_loss += 1 * sum(list(dict_view_loss.values())) + \
                       1 * sum((list(dict_level_fusion_loss.values()))) + \
                       1 * loss_pre
            # print(1)
        # return [i[-1] for i in list(dict_view_feat.values())] + [MMfeature, pre], ci_loss, dict_view_att_score
        return [i[-1] for i in list(dict_view_feat.values())] + [MMfeature, pre], ci_loss, dict_view_graph
