import numpy as np
from numba import njit


def binary_f1_score(conf_matrix):
    # 计算二分类混淆矩阵的F1分数
    f1_score = 0
    for label in (0, 1):
        if label == 0:
            tp, fp, fn, _ = conf_matrix
        else:
            _, fn, fp, tp = conf_matrix
        if (tp + fp) == 0 or (tp + fn) == 0:
            return -np.inf
        # 计算精确率和召回率
        pr = tp / (tp + fp)
        re = tp / (tp + fn)
        if (pr + re) == 0:
            return -np.inf
        # 计算F1分数
        f1 = 2 * (pr * re) / (pr + re)
        f1_score += f1
    # 返回两个类别F1分数的平均值
    return f1_score / 2