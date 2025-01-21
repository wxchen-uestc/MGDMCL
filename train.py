import numpy as np
import utils
import torch
from MGDMCL import FUSION
import ModelEvaluate
import pandas as pd
import time

cuda = True if torch.cuda.is_available() else False


def data_prepare(data_fold):
    # choice omics
    omics = [1, 2, 3]  # mRNA, miRNA, DNA
    # training set
    x_tr = [np.loadtxt(data_fold + r'\{}_tr.csv'.format(i), delimiter=',') for i in omics]
    y_tr = np.loadtxt(data_fold + r'\labels_tr.csv', delimiter=',').astype(int)
    # validation set or testing set
    x_te = [np.loadtxt(data_fold + r'\{}_te.csv'.format(i), delimiter=',') for i in omics]
    y_te = np.loadtxt(data_fold + r'\labels_te.csv', delimiter=',').astype(int)

    num_class = int(max(y_tr) + 1)
    x_trte = [np.concatenate([x_tr[i], x_te[i]]) for i in range(len(omics))]
    sample_weight = utils.cal_sample_weight(y_tr, num_class)
    # calculate matrix_same
    matrix_same = []
    for i in range(len(y_tr)):
        tmp = y_tr == y_tr[i]
        matrix_same.append(list(tmp.astype(float)))
    matrix_same = np.array(matrix_same)
    if cuda:
        for i in range(len(x_tr)):
            x_tr[i] = torch.FloatTensor(x_tr[i]).cuda()
            x_trte[i] = torch.FloatTensor(x_trte[i]).cuda()
        y_tr = torch.LongTensor(y_tr).cuda()
        sample_weight = torch.FloatTensor(sample_weight).cuda()
        matrix_same = torch.FloatTensor(matrix_same).cuda()
        mask_anchor_out = (1 - torch.eye(matrix_same.shape[0])).cuda()
        matrix_same = matrix_same * mask_anchor_out
    return x_tr, y_tr, sample_weight, x_trte, y_te, matrix_same


def Train(x, y, sample_weight, model, optim, matrix_same):
    model.train()
    optim.zero_grad()
    # train
    pre, ci_loss, dict_view_graph = model(x, y, sample_weight, matrix_same, Training=True)

    ci_loss.backward()
    optim.step()


def Test(x_trte, y_te, model):
    model.eval()
    with torch.no_grad():
        pre, ci_loss, dict_view_graph = model(x_trte)
        pre = pre[-1]
        pre = pre.detach().cpu()[-len(y_te):]
    return ModelEvaluate.get_result(pre, y_te)


def Train_Test(data_fold, li_hid_dim, k, mask_rate, a, b, lr, epochs):
    # get data
    x_tr, y_tr, sample_weight, x_trte, y_te, matrix_same = data_prepare(data_fold)
    #
    num_class = int(max(y_tr) + 1)
    in_dim = [x.shape[1] for x in x_tr]
    # define model and optimal
    model = FUSION(in_dim=in_dim, li_hid_dim=li_hid_dim,
                   k=k, num_class=num_class, dropout=0.5, mask_rate=mask_rate, a=a, b=b)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    if cuda:
        model = model.cuda()
    # train
    result = []
    for epoch in range(epochs):
        Train(x_tr, y_tr, sample_weight, model, optim, matrix_same)
        if epoch % 50 == 0:
            result.append(Test(x_trte, y_te, model))
    # Test
    result.append(Test(x_trte, y_te, model))


Train_Test(data_fold=r'D:\通用分类\数据\划分数据\BRCA',
           li_hid_dim=[300, 300, 300],
           k=10,
           mask_rate=0.6,
           a=0.01,
           b=0.01,
           lr=1e-4,
           epochs=10000)

