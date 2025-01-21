import torch
import torch.nn as nn
import torch.nn.functional as F
from model.common_layer import LinearLayer, Graph, GateSelect
import numpy as np
import pandas as pd
import utils


def encoding_mask_noise_by_label(x, y, mask_rate=0.3):
    y_pd = pd.DataFrame(y.cpu().numpy())
    label_loc = [group.index.to_numpy() for _, group in y_pd.groupby(0)]
    mask_nodes, keep_nodes = [], []
    for _ in label_loc:
        #
        num_nodes = len(_)
        perm = torch.randperm(num_nodes)
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        mask_nodes.append(_[perm[: num_mask_nodes]])
        keep_nodes.append(_[perm[num_mask_nodes:]])

    mask_nodes = np.concatenate(mask_nodes)
    keep_nodes = np.concatenate(keep_nodes)

    out_x = x.clone()
    out_x[mask_nodes] = 0.0
    return out_x, (mask_nodes, keep_nodes)


class AglMGae(nn.Module):
    def __init__(self,
                 encoder_li_dim,
                 encoder_flag_graph,
                 encoder_head_num,
                 encoder_flag_res,
                 encoder_flag_normalize,
                 encoder_activate_fun,
                 k=10,
                 dropout=0.5,
                 mask_rate=0.6,
                 decoder_li_dim=None,
                 decoder_flag_graph=None,
                 decoder_head_num=None,
                 decoder_flag_res=None,
                 decoder_flag_normalize=None,
                 decoder_activate_fun=None):
        super().__init__()

        # Encoder
        self.gate = GateSelect(encoder_li_dim[0], dropout)
        self.encoder = Graph(encoder_li_dim, encoder_flag_graph, encoder_head_num, encoder_flag_res,
                             encoder_flag_normalize, encoder_activate_fun)

        self.encoder_to_decoder = LinearLayer(encoder_li_dim[-1], decoder_li_dim[0])
        self.decoder = Graph(decoder_li_dim, decoder_flag_graph, decoder_head_num, decoder_flag_res,
                             decoder_flag_normalize, decoder_activate_fun)

        self.k = k
        self.mask_rate = mask_rate

    def forward(self, x, y=None, Training=False):
        att_score, feat_emb = self.gate(x)
        feat_emb_norm = F.normalize(feat_emb, dim=0)
        #
        adj_gen = utils.get_adj(feat_emb_norm, self.k)
        #
        edge_index, edge_weight = utils.to_sparse(adj_gen)
        # GCN
        feat_emb_gcn = self.encoder(feat_emb_norm, edge_index, edge_weight)
        loss = {}
        if Training:
            #
            feat_emb_gcn_mask, (mask_nodes, keep_nodes) = encoding_mask_noise_by_label(feat_emb_gcn[-1], y, self.mask_rate)
            feat_emb_de = self.encoder_to_decoder(feat_emb_gcn_mask)
            feat_emb_gcn_de = self.decoder(feat_emb_de, edge_index, edge_weight)
            loss['loss_recon'] = utils.sce_loss(feat_emb_gcn_de[-1][mask_nodes], feat_emb_norm[mask_nodes])
            loss['loss_att'] = torch.mean(att_score)
        return feat_emb_gcn, loss, adj_gen.cpu()

