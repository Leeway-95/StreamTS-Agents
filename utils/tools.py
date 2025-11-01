import numpy as np
from numba import njit, objmode
from utils.config import *


def _rolling_knn(dists, knns, dist, knn, knn_insert_idx, knn_fill, l, k_neighbours, lbound):
    # 增量更新KNN矩阵和距离
    dists[knn_insert_idx, :] = dist[knn]
    knns[knn_insert_idx, :] = knn
    idx = np.arange(lbound, l)
    change_mask = np.full(shape=l - lbound, fill_value=True, dtype=np.bool_)
    for kdx in range(k_neighbours - 1):
        change_idx = dist[idx] < dists[idx, kdx]
        change_idx = np.logical_and(change_idx, change_mask[idx - lbound])
        change_idx = idx[change_idx]
        change_mask[change_idx - lbound] = False
        knns[change_idx, kdx + 1:] = knns[change_idx, kdx:k_neighbours - 1]
        knns[change_idx, kdx] = knn_insert_idx
        dists[change_idx, kdx + 1:] = dists[change_idx, kdx:k_neighbours - 1]
        dists[change_idx, kdx] = dist[change_idx]
    lbound = max(0, lbound - 1)
    knn_fill = min(knn_fill + 1, knn_insert_idx)
    return knns, dists, lbound, knn_fill


def _sliding_dot(query, time_series):
    # 使用FFT卷积计算滑动点积
    m = len(query)
    n = len(time_series)
    time_series_add = 0
    if n % 2 == 1:
        time_series = np.concatenate((np.array([0]), time_series))
        time_series_add = 1
    q_add = 0
    if m % 2 == 1:
        query = np.concatenate((np.array([0]), query))
        q_add = 1
    query = query[::-1]
    query = np.concatenate((query, np.zeros(n - m + time_series_add - q_add)))
    trim = m - 1 + time_series_add
    with objmode(dot_product="float64[:]"):
        dot_product = np.fft.irfft(np.fft.rfft(time_series) * np.fft.rfft(query))
    return dot_product[trim:]


def _argkmin(dist, k, lbound):
    # 查找距离数组中k个最小值的索引
    args = np.zeros(shape=k, dtype=np.int64)
    vals = np.zeros(shape=k, dtype=np.float64)
    for idx in range(k):
        min_arg = np.nan
        min_val = np.inf
        for kdx in range(lbound, dist.shape[0]):
            val = dist[kdx]
            if val < min_val:
                min_val = val
                min_arg = kdx
        min_arg = np.int64(min_arg)
        args[idx] = min_arg
        vals[idx] = min_val
        dist[min_arg] = np.inf
    dist[args] = vals
    return args


def _roll_numba(arr, num, fill_value=0):
    # 使用Numba优化的数组滚动操作
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    else:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    return result


def _mean(idx, csum, window_size):
    # 从累积和计算滚动平均值
    window_sum = csum[idx + window_size] - csum[idx]
    return window_sum / window_size


def _std(idx, csumsq, csum, window_size):
    # 从累积和计算滚动标准差
    window_sum = csum[idx + window_size] - csum[idx]
    window_sum_sq = csumsq[idx + window_size] - csumsq[idx]
    movstd = window_sum_sq / window_size - (window_sum / window_size) ** 2
    if movstd < 0:
        return 1
    movstd = np.sqrt(movstd)
    if abs(movstd) < 1e-3:
        return 1
    return movstd


def moving_mean(ts, w):
    # 计算窗口大小为w的移动平均值
    cumsum = np.cumsum(ts)
    moving_avg = np.zeros(len(ts) - w + 1)
    moving_avg[0] = cumsum[w - 1] / w
    for i in range(1, len(ts) - w + 1):
        moving_avg[i] = (cumsum[i + w - 1] - cumsum[i - 1]) / w
    return moving_avg


def rank_binary_data(data):
    # 计算二元数据类别的平均排名
    zeros = data == 0
    ones = data == 1
    zero_ranks = np.arange(np.sum(zeros))
    one_ranks = np.arange(zero_ranks.shape[0], data.shape[0])
    zero_mean = np.mean(zero_ranks) + 1 if zero_ranks.shape[0] > 0 else 0
    one_mean = np.mean(one_ranks) + 1 if one_ranks.shape[0] > 0 else 0
    ranks = np.full(data.shape[0], fill_value=zero_mean, dtype=np.float64)
    ranks[ones] = one_mean
    return ranks


def _labels(knn, split_idx):
    # 为KNN分类生成标签
    n_timepoints, k_neighbours = knn.shape
    y_true = np.concatenate((
        np.zeros(split_idx, dtype=np.int32),
        np.ones(n_timepoints - split_idx, dtype=np.int32),
    ))
    knn_labels = np.zeros(shape=(k_neighbours, n_timepoints), dtype=np.int32)
    for i_neighbor in range(k_neighbours):
        neighbours = knn[:, i_neighbor]
        knn_labels[i_neighbor] = y_true[neighbours]
    ones = np.sum(knn_labels, axis=0)
    zeros = k_neighbours - ones
    y_pred = np.asarray(ones > zeros, dtype=np.int32)
    return y_true, y_pred

def valid_call(task, dataset, method):
    if task == 'UNDERSTANDING':
        if dataset in DATASET_UNDERSTANDING and method in BASELINE_UNDERSTANDING:
            return True
        else:
            return False
    elif  task == 'REASONING':
        if dataset in DATASET_REASONING and method in BASELINE_REASONING:
            return True
        else:
            return False
    elif  task == 'FORECASTING_NUM':
        if dataset in DATASET_FORECASTING_NUM and method in BASELINE_FORECASTING_NUM:
            return True
        else:
            return False
    elif  task == 'FORECASTING_EVENT':
        if dataset in DATASET_FORECASTING_EVENT and method in BASELINE_FORECASTING_EVENT:
            return True
        else:
            return False