import os
import numpy as np
import torch
import sys
import torch.nn.functional as F

cuda = True if torch.cuda.is_available() else False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def cal_sample_weight(labels, num_class, use_sample_weight=True):
    if not use_sample_weight:
        return np.ones(len(labels)) / len(labels)

    count_dict = dict()
    for i in range(num_class):
        count_dict["{:}".format(i)] = np.sum(labels == i)

    sort_count_dict_Asc = dict(sorted(count_dict.items(), key=lambda x: x[1]))
    sort_count_dict_Des = dict(sorted(count_dict.items(), key=lambda x: x[1], reverse=True))

    count = list(sort_count_dict_Asc.values())
    sample_weight = np.zeros(labels.shape)
    for i in range(num_class):
        num = int(list(sort_count_dict_Des.keys())[i])
        new_count = list(sort_count_dict_Asc.values())[i]
        sample_weight[np.where(labels == num)[0]] = new_count / np.sum(count)

    return sample_weight


def one_hot_tensor(y, num_dim):
    y_onehot = torch.zeros(y.shape[0], num_dim)
    y_onehot.scatter_(1, y.view(-1, 1), 1)

    return y_onehot


def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def to_sparse(x):
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)
    indices = torch.nonzero(x)
    if len(indices.shape) == 0:
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return indices, values


def cal_adj_mat_parameter(edge_per_node, data, metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    dist = cosine_distance_torch(data, data)
    parameter = torch.sort(dist.reshape(-1, )).values[edge_per_node * data.shape[0]]
    return np.ndarray.item(parameter.data.cpu().numpy())


def graph_from_dist_tensor(dist, parameter, self_dist=True):
    if self_dist:
        assert dist.shape[0] == dist.shape[1], "Input is not pairwise dist matrix"
    g = (dist <= parameter).float()
    if self_dist:
        diag_idx = np.diag_indices(g.shape[0])
        g[diag_idx[0], diag_idx[1]] = 0

    return g


def gen_adj_mat_tensor(data, parameter, metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    dist = cosine_distance_torch(data, data)
    g = graph_from_dist_tensor(dist, parameter, self_dist=True)
    if metric == "cosine":
        adj = 1 - dist
    else:
        raise NotImplementedError
    adj = adj * g
    adj_T = adj.transpose(0, 1)
    I = torch.eye(adj.shape[0])
    if cuda:
        I = I.cuda()
    adj = adj + adj_T * (adj_T > adj).float() - adj * (adj_T > adj).float()
    adj = adj + I

    return adj


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    D = torch.sum(adj, -1)
    d_inv_sqrt = torch.pow(D, -0.5)
    d_inv_sqrt = torch.diagflat(d_inv_sqrt)
    adj = torch.mm(d_inv_sqrt, torch.mm(adj, d_inv_sqrt))
    return adj


def save_model_dict(folder, model_dict):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for module in model_dict:
        torch.save(model_dict[module].state_dict(), os.path.join(folder, module + ".pth"))


def load_model_dict(folder, model_dict):
    for module in model_dict:
        if os.path.exists(os.path.join(folder, module + ".pth")):
            #            print("Module {:} loaded!".format(module))
            model_dict[module].load_state_dict(torch.load(os.path.join(folder, module + ".pth"),
                                                          map_location="cuda:{:}".format(torch.cuda.current_device())))
        else:
            print("WARNING: Module {:} from model_dict is not loaded!".format(module))
        if cuda:
            model_dict[module].cuda()
    return model_dict


def SupervisedContrastiveLoss(projections, targets, temperature):
    device_ = torch.device("cuda") if projections.is_cuda else torch.device("cpu")
    dot_product_tempered = torch.mm(projections, projections.T) / temperature
    exp_dot_tempered = torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5

    # Mask for similar classes (positive pairs)
    mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device_)

    # Mask to exclude anchor itself
    mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device_)

    # Combined mask for positive pairs (same class and not anchor itself)
    mask_combined = mask_similar_class * mask_anchor_out

    # Cardinality per sample (number of positive pairs per sample)
    cardinality_per_samples = torch.sum(mask_combined, dim=1)

    # Avoid division by zero
    cardinality_per_samples[cardinality_per_samples == 0] = 1

    # Mask for different classes (negative pairs)
    mask_different_class = ~mask_similar_class

    # Log probabilities for positive pairs
    numerator = torch.sum(exp_dot_tempered * mask_combined, dim=1)
    denominator = torch.sum(exp_dot_tempered * (mask_anchor_out + mask_different_class), dim=1)
    log_prob = -torch.log(numerator / denominator)

    # Supervised Contrastive Loss per sample
    supervised_contrastive_loss_per_sample = log_prob / cardinality_per_samples

    # Mean Supervised Contrastive Loss over all samples
    supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)

    return supervised_contrastive_loss
    # return supervised_contrastive_loss


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


def get_adj(x, k):
    # 计算阈值
    p = cal_adj_mat_parameter(k, x)
    # 生成PSN矩阵
    adj_gen = gen_adj_mat_tensor(x, p)
    # 归一化
    adj_gen = normalize_adj(adj_gen)
    return adj_gen