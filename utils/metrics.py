#https://github.com/lquirosd/P2PaLA/blob/6c1f1cb8d19af26f4590b0f07fdbe513858e2235/evalTools/metrics.py#L270  
from __future__ import print_function
from __future__ import division
from builtins import range

import os
import sys

import numpy as np

import matplotlib.pyplot as plt
import time
from utils.bounding_box import BoundingBox
from utils.coco_evaluator import BBFormat, BBType, CoordinatesType, get_coco_summary

def timing_val(func):
    def wrapper(*arg, **kw):
        '''source: http://www.daniweb.com/code/snippet368.html'''
        t1 = time.time()
        res = func(*arg, **kw)
        t2 = time.time()
        print(f' {func.__name__} took {(t2 - t1)} seconds')
        return res
    return wrapper

# @timing_val
def calc_detection_metric(hyps, targets, min_th=0.5):
    ret_gt, ret_hyp = [], []
    # obj_class = 'textline'
    img_size = (768,1024)
    i_total = 0
    for i, target in enumerate(hyps):
        i_total += 1
        v = target['instances'].get_fields()
        scores = v['scores']
        classes = v['pred_classes']
        pred_boxes = v['pred_boxes']
        img_name = f'image_{i_total}'
        
        for j, pred_box_ in enumerate(pred_boxes):
            x0, y0, x1, y1 = pred_box_
            score = scores[j]
            c = classes[j]
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            if score > min_th:
                bb = BoundingBox(image_name=img_name,
                                        class_id=c,
                                        coordinates=(x0, y0, x1, y1),
                                        img_size=img_size,
                                        bb_type=BBType.DETECTED,
                                        confidence=1.0,
                                        type_coordinates=CoordinatesType.ABSOLUTE,
                                        format=BBFormat.XYX2Y2)
                ret_hyp.append(bb)
        
        v = targets[i]['instances'].get_fields()
        pred_boxes = v['gt_boxes']
        gt_classes = v['gt_classes']
        for j, [x0, y0, x1, y1] in enumerate(pred_boxes):
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            c = gt_classes[j]
            bb = BoundingBox(image_name=img_name,
                                    class_id=c, # TODO redo class GT
                                    coordinates=(x0, y0, x1, y1),
                                    img_size=img_size,
                                    # confidence=1.0,
                                    type_coordinates=CoordinatesType.ABSOLUTE,
                                    bb_type=BBType.GROUND_TRUTH,
                                    format=BBFormat.XYX2Y2)
            ret_gt.append(bb)
    res = get_coco_summary(ret_gt, ret_hyp)
    
    return res

def calc_metrics(hyps, targets):
    pix_accs, mean_accs, mean_ius, freq_w_ius = [], [], [], []
    for i, hyp in enumerate(hyps):
        gt = targets[i]
        pix_acc = pixel_accuraccy(hyp, gt)
        mean_acc = mean_accuraccy(hyp, gt)
        mean_iu = mean_IU(hyp, gt)
        freq_w_iu = freq_weighted_IU(hyp, gt)
        pix_accs.append(pix_acc)
        mean_accs.append(mean_acc)
        mean_ius.append(mean_iu)
        freq_w_ius.append(freq_w_iu)
    return pix_accs, mean_accs, mean_ius, freq_w_ius
# --- Pixel level accuraccy
def pixel_accuraccy(hyp, target):
    """
    computes pixel by pixel accuraccy: sum_i(n_ii)/sum_i(t_i)
    """
    return (target == hyp).sum() / target.size


# -- mean accuraccy
def mean_accuraccy(hyp, target):
    """
    computes mean accuraccy: 1/n_cl * sum_i(n_ii/t_i)
    """
    s, cl = per_class_accuraccy(hyp, target)
    return np.sum(s) / cl.size


def jaccard_index(hyp, target):
    """
    computes jaccard index (I/U)
    """
    smooth = np.finfo(np.float).eps
    cl = np.unique(target)
    n_cl = cl.size
    j_index = np.zeros(n_cl)
    for i, c in enumerate(cl):
        I = (target[target == hyp] == c).sum()
        U = (target == c).sum() + (hyp == c).sum()
        j_index[i] = (I + smooth) / (U - I + smooth)
    return (j_index, cl)


def mean_IU(hyp, target):
    """
    computes mean IU as defined in https://arxiv.org/pdf/1411.4038.pdf
    """
    j_index, cl = jaccard_index(hyp, target)
    return np.sum(j_index) / cl.size


def freq_weighted_IU(hyp, target):
    """
    computes frequency weighted IU as defined in https://arxiv.org/pdf/1411.4038.pdf
    """
    j_index, cl = jaccard_index(hyp, target)
    _, n_cl = np.unique(target, return_counts=True)
    return np.sum(j_index * n_cl) / target.size

def per_class_accuraccy(hyp, target):
    """
    computes pixel by pixel accuraccy per class in target
    sum_i(n_ii/t_i)
    """
    cl = np.unique(target)
    n_cl = cl.size
    s = np.zeros(n_cl)
    # s[0] = (target[target==hyp]==0).sum()/(target==0).sum()
    for i, c in enumerate(cl):
        s[i] = (target[target == hyp] == c).sum() / (target == c).sum()
    return (s, cl)