import torch
import numpy as np
from collections import OrderedDict
import pandas as pd
import os
from tqdm import tqdm
import cv2
from utils.misc import split_np_imgrid, get_np_imgrid


def cal_ber(tn, tp, fn, fp):
    return  0.5*(fp/(tn+fp) + fn/(fn+tp))

def cal_acc(tn, tp, fn, fp):
    return (tp + tn) / (tp + tn + fp + fn)


def get_binary_classification_metrics(pred, gt, threshold=None):
    if threshold is not None:
        gt = (gt > threshold)
        pred = (pred > threshold)
    TP = np.logical_and(gt, pred).sum()
    TN = np.logical_and(np.logical_not(gt), np.logical_not(pred)).sum()
    FN = np.logical_and(gt, np.logical_not(pred)).sum()
    FP = np.logical_and(np.logical_not(gt), pred).sum()
    BER = cal_ber(TN, TP, FN, FP)
    ACC = cal_acc(TN, TP, FN, FP)
    return OrderedDict( [('TP', TP),
                        ('TN', TN),
                        ('FP', FP),
                        ('FN', FN),
                        ('BER', BER),
                        ('ACC', ACC)]
                      )


def evaluate(res_root, pred_id, gt_id, nimg, nrow, threshold):
    img_names  = os.listdir(res_root)
    score_dict = OrderedDict()

    for img_name in img_names:
        im_grid_path = os.path.join(res_root, img_name)
        im_grid = cv2.imread(im_grid_path)
        ims = split_np_imgrid(im_grid, nimg, nrow)
        pred = ims[pred_id]
        gt = ims[gt_id]
        score_dict[img_name] = get_binary_classification_metrics(pred,
                                                                 gt,
                                                                 threshold)
            
    df = pd.DataFrame(score_dict)
    df['ave'] = df.mean(axis=1)

    tn = df['ave']['TN']
    tp = df['ave']['TP']
    fn = df['ave']['FN']
    fp = df['ave']['FP']

    pos_err = (1 - tp / (tp + fn)) * 100
    neg_err = (1 - tn / (tn + fp)) * 100
    ber = (pos_err + neg_err) / 2
    acc = (tn + tp) / (tn + tp + fn + fp)

    return pos_err, neg_err, ber, acc, df







###############################################

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, val, weight=1):
        self.sum += val * weight
        self.count += weight

    def average(self):
        if self.count == 0:
            return 0
        else:
            return self.sum / self.count

    def clear(self):
        self.sum = 0
        self.count = 0

def compute_cm_torch(y_pred, y_label, n_class):
    mask = (y_label >= 0) & (y_label < n_class)
    hist = torch.bincount(n_class * y_label[mask] + y_pred[mask],
                          minlength=n_class**2).reshape(n_class, n_class)
    return hist

class MyConfuseMatrixMeter(AverageMeter):
    """More Clear Confusion Matrix Meter"""
    def __init__(self, n_class):
        super(MyConfuseMatrixMeter, self).__init__()
        self.n_class = n_class

    def update_cm(self, y_pred, y_label, weight=1):
        y_label = y_label.type(torch.int64)
        val = compute_cm_torch(y_pred=y_pred.flatten(), y_label=y_label.flatten(),
                               n_class=self.n_class)
        self.update(val, weight)

    # def get_scores_binary(self):
    #     assert self.n_class == 2, "this function can only be called for binary calssification problem"
    #     tn, fp, fn, tp = self.sum.flatten()
    #     eps = torch.finfo(torch.float32).eps
    #     precision = tp / (tp + fp + eps)
    #     recall = tp / (tp + fn + eps)
    #     f1 = 2*recall*precision / (recall + precision + eps)
    #     iou = tp / (tp + fn + fp + eps)
    #     oa = (tp + tn) / (tp + tn + fn + fp + eps)
    #     score_dict = {}
    #     score_dict['precision'] = precision.item()
    #     score_dict['recall'] = recall.item()
    #     score_dict['f1'] = f1.item()
    #     score_dict['iou'] = iou.item()
    #     score_dict['oa'] = oa.item()
    #     return score_dict
    def get_scores_binary(self):
        assert self.n_class == 2, "this function can only be called for binary calssification problem"
        tn, fp, fn, tp = self.sum.flatten()
        eps = torch.finfo(torch.float32).eps
        pos_err = (1 - tp / (tp + fn + eps)) * 100
        neg_err = (1 - tn / (tn + fp + eps)) * 100
        ber = (pos_err + neg_err) / 2
        acc = (tn + tp) / (tn + tp + fn + fp + eps)
        score_dict = {}
        score_dict['pos_err'] = pos_err
        score_dict['neg_err'] = neg_err
        score_dict['ber'] = ber
        score_dict['acc'] = acc
        return score_dict
