"""
代码主题
①  注意多分类 返回AUC, ACC, f1_W, f1_M, 二分类返回 AUC, ACC, f1
代码问题
① 多分类的AUC的参数如何选取 44行


"""
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import torch


def get_result(prediction_proba, labels):
    prediction = prediction_proba.argmax(1)
    # 注意: 由于采用分层采样的方式，所以保证labels中具有所有的样本类别
    num_class = int(max(labels)) + 1
    if num_class == 2:
        # AUC
        AUC = roc_auc_score(labels, torch.softmax(torch.FloatTensor(prediction_proba), dim=1)[:, 1])
        # ACC
        ACC = accuracy_score(labels, prediction)
        # F1
        f1 = f1_score(labels, prediction)
        # pre
        pre = precision_score(labels, prediction)
        # recall
        recall = recall_score(labels, prediction)
        # print("AUC:{:.3f}, ACC:{:.3f}, F1:{:.3f} Pre:{:.3f} Recall:{:.3f}".format(AUC, ACC, f1, pre, recall))
        print("{:.3f} {:.3f} {:.3f}".format(AUC, ACC, f1))
        return AUC, ACC, f1, pre, recall
    if num_class > 2:
        # ACC
        ACC = accuracy_score(labels, prediction)
        # F1-W
        f1_W = f1_score(labels, prediction, average='weighted')
        # F1-M
        f1_M = f1_score(labels, prediction, average='macro')
        # print("ACC:{:.3f}, F1-W:{:.3f}, F1-M:{:.3f}".format(ACC, f1_W, f1_M))
        print("{:.3f} {:.3f} {:.3f}".format(ACC, f1_W, f1_M))
        return ACC, f1_W, f1_M