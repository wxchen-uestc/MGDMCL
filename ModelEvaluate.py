from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import torch


def get_result(prediction_proba, labels):
    prediction = prediction_proba.argmax(1)
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
        print("AUC:{:.3f} ACC:{:.3f} F1:{:.3f}".format(AUC, ACC, f1))
        return AUC, ACC, f1, pre, recall
    if num_class > 2:
        # ACC
        ACC = accuracy_score(labels, prediction)
        # F1-W
        f1_W = f1_score(labels, prediction, average='weighted')
        # F1-M
        f1_M = f1_score(labels, prediction, average='macro')
        # print("ACC:{:.3f}, F1-W:{:.3f}, F1-M:{:.3f}".format(ACC, f1_W, f1_M))
        print("ACC:{:.3f} F1_W:{:.3f} F1_M:{:.3f}".format(ACC, f1_W, f1_M))
        return ACC, f1_W, f1_M