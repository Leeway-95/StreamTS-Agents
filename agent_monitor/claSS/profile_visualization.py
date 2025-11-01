import matplotlib
import numpy as np

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt

# 设置matplotlib样式，替代seaborn
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def plot_profile(ts_name, profile, true_cps=None, found_cps=None, show=True, score="Score", save_path=None,
                 font_size=26):
    plt.clf()
    fig, ax = plt.subplots(1, figsize=(20, 5))

    ax.plot(np.arange(profile.shape[0]), profile, color='b')

    ax.set_title(ts_name, fontsize=font_size)
    ax.set_xlabel('split point  $s$', fontsize=font_size)
    ax.set_ylabel(score, fontsize=font_size)

    if true_cps is not None:
        for i, true_cp in enumerate(true_cps):
            ax.axvline(x=true_cp, linewidth=2, color='r', linestyle='--', label='True Change Point' if i == 0 else None)

    if found_cps is not None:
        for i, found_cp in enumerate(found_cps):
            ax.axvline(x=found_cp, linewidth=2, color='g', linestyle='--', label='Found Change Point' if i == 0 else None)

    if true_cps is not None or found_cps is not None:
        plt.legend(prop={'size': font_size})

    if show is True:
        plt.show()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")


def plot_ts(ts_name, ts, true_cps=None, show=True, save_path=None, font_size=26):
    plt.clf()
    fig, ax = plt.subplots(1, figsize=(20, 5))

    if true_cps is not None:
        segments = [0] + true_cps.tolist() + [ts.shape[0]]
        for idx in np.arange(0, len(segments) - 1):
            ax.plot(np.arange(segments[idx], segments[idx + 1]), ts[segments[idx]:segments[idx + 1]])
    else:
        ax.plot(np.arange(ts.shape[0]), ts)

    ax.set_title(ts_name, fontsize=font_size)

    # if true_cps is not None:
    # ax.legend(prop={'size': font_size})

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(font_size)

    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(font_size)

    if show is True:
        plt.show()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")


def plot_profile_with_ts(ts_name, ts, profile, true_cps=None, found_cps=None, show=True, score="Score", save_path=None,
                         font_size=26):
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, gridspec_kw={'hspace': .05}, figsize=(20, 10))

    if true_cps is not None:
        segments = [0] + true_cps.tolist() + [ts.shape[0]]
        for idx in np.arange(0, len(segments) - 1):
            ax1.plot(np.arange(segments[idx], segments[idx + 1]), ts[segments[idx]:segments[idx + 1]])
    else:
        ax1.plot(np.arange(ts.shape[0]), ts)
    
    # 始终绘制完整的连续profile线段
    ax2.plot(np.arange(profile.shape[0]), profile, color='b', linewidth=1.5)

    ax1.set_title(ts_name, fontsize=font_size)
    ax2.set_xlabel('split point  $s$', fontsize=font_size)
    ax2.set_ylabel(score, fontsize=font_size)

    for ax in (ax1, ax2):
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(font_size)

        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(font_size)

    if true_cps is not None:
        for idx, true_cp in enumerate(true_cps):
            ax1.axvline(x=true_cp, linewidth=2, color='r', linestyle='--', label=f'True Change Point' if idx == 0 else None)
            ax2.axvline(x=true_cp, linewidth=2, color='r', linestyle='--', label='True Change Point' if idx == 0 else None)

    if found_cps is not None:
        for idx, found_cp in enumerate(found_cps):
            ax1.axvline(x=found_cp, linewidth=2, color='g', linestyle='--', label='Predicted Change Point' if idx == 0 else None)
            ax2.axvline(x=found_cp, linewidth=2, color='g', linestyle='--', label='Predicted Change Point' if idx == 0 else None)

    ax1.legend(prop={'size': font_size})

    if show is True:
        plt.show()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")


def plot_single_segment(segment_data, segment_name, save_path=None, color='#267BB6'):
    """
    绘制单个时间序列段，无网格无坐标系
    
    Args:
        segment_data: 时间序列数据
        segment_name: 段名称
        save_path: 保存路径
        color: 线条颜色
    """
    plt.clf()
    # 设置图片尺寸为正方形，确保输出512x512像素
    fig, ax = plt.subplots(1, figsize=(5.12, 5.12))
    
    # 绘制时间序列，使用指定颜色的实线
    ax.plot(np.arange(len(segment_data)), segment_data, color=color, linewidth=2, linestyle='-')
    
    # 移除网格和坐标系
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # 设置标题
    ax.set_title(segment_name, fontsize=16, pad=20)
    
    if save_path is not None:
        # 设置DPI为100，配合5.12x5.12英寸的图片尺寸，得到512x512像素
        plt.savefig(save_path, bbox_inches="tight", dpi=100)
        plt.close()
