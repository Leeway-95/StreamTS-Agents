"""
文件夹管理模块
包含创建和管理各种文件夹的功能
"""

import os


def create_images_folder(cnt_original_pattern_sample):
    """
    创建images文件夹（如果不存在）
    """
    if not os.path.exists(f'pattern_{cnt_original_pattern_sample}'):
        os.makedirs(f'pattern_{cnt_original_pattern_sample}')
        print("创建了images文件夹")
    return True


def create_samples_folder():
    """
    创建images_samples文件夹（如果不存在）
    """
    if not os.path.exists('pattern_samples'):
        os.makedirs('pattern_samples')
        print("创建了images_samples文件夹")
    return True


def create_no_pattern_samples_folder():
    """
    创建no_pattern_samples文件夹（如果不存在）
    """
    if not os.path.exists('no_pattern_samples'):
        os.makedirs('no_pattern_samples')
        print("创建了no_pattern_samples文件夹")
    return True


def create_pattern_folders(dataset_dir):
    """
    创建9个模式分类文件夹（如果不存在）
    """
    pattern_folders = {
        'pattern_sudden_spike_outlier': 'Sudden Spike Outlier',
        'pattern_level_shift_outlier': 'Level Shift Outlier',
        'pattern_upward_trend': 'Upward Trend',
        'pattern_downward_trend': 'Downward Trend',
        'pattern_fixed_seasonality': 'Fixed Seasonality',
        'pattern_shifting_seasonality': 'Shifting Seasonality',
        'pattern_obvious_volatility': 'Obvious Volatility',
        'no_temporal_pattern': 'No Temporal Pattern'
    }

    # 首先创建dataset目录
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # 在dataset目录下创建各个模式文件夹
    for folder_name in pattern_folders.keys():
        full_path = os.path.join(dataset_dir, folder_name)
        if not os.path.exists(full_path):
            os.makedirs(full_path)

    return pattern_folders