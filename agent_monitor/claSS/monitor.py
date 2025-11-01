import ast
import os
import sys
import pandas as pd
import numpy as np

from agent_monitor.claSS.utils import parse_series, parse_positions, segment_time_series, process_dataset, create_results_csv
from profile_visualization import plot_profile_with_ts, plot_profile, plot_ts, plot_single_segment

def visualize_sample_data(csv_path='stream_summary.csv'):
    """
    可视化所有样本的分割结果
    
    Args:
        csv_path: CSV文件路径
    """
    
    if not os.path.exists(csv_path):
        print(f"数据文件不存在: {csv_path}")
        return
    
    print(f"\n{'='*60}")
    print("开始生成可视化图表")
    print(f"{'='*60}")
    
    # 创建输出目录
    output_dir = "vis_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 读取数据
    df = pd.read_csv(csv_path, sep=',')
    
    if len(df) == 0:
        print("数据文件为空")
        return
    
    print(f"数据文件包含 {len(df)} 个样本")
    
    # 处理所有样本
    success_count = 0
    for idx, row in df.iterrows():
        sample_index = row['Index']
        print(f"\n处理样本 {sample_index}...")
        
        try:
            # 解析数据
            series = parse_series(row['Series'])
            true_segments = parse_positions(row['Positions'])
            
            if len(series) == 0 or len(true_segments) == 0:
                print(f"  样本 {sample_index} 数据无效，跳过")
                continue
            
            print(f"  时间序列长度: {len(series)}")
            print(f"  真实分割数量: {len(true_segments)}")
            
            # 进行分割并获取profile
            predicted_segments, profile = segment_time_series(series, return_profile=True)
            print(f"  预测分割数量: {len(predicted_segments)}")
            
            # 转换分割点格式用于可视化
            true_cps = []
            for start, end in true_segments:
                if start > 0:  # 不包括起始点0
                    true_cps.append(start)
            true_cps = np.array(true_cps) if true_cps else None
            
            found_cps = []
            for start, end in predicted_segments:
                if start > 0:  # 不包括起始点0
                    found_cps.append(start)
            found_cps = np.array(found_cps) if found_cps else None
            
            # 生成可视化图表
            ts_name = f"Streaming Time Series - {sample_index}"
            
            if np.any(profile != 0):  # 只有当profile有有效数据时才绘制
                save_path_combined = os.path.join(output_dir, f"Series_{sample_index}.png")
                plot_profile_with_ts(ts_name, series, profile, true_cps=true_cps,
                                   found_cps=found_cps, show=False, save_path=save_path_combined)
                print(f"  保存图表: {save_path_combined}")
                
                # 创建子目录存放分割后的时间序列图片
                series_dir = os.path.join(output_dir, f"Series_{sample_index}")
                if not os.path.exists(series_dir):
                    os.makedirs(series_dir)
                    print(f"  创建子目录: {series_dir}")
                
                # 生成每个分割段的时间序列图片 - 使用预测分割结果
                segment_count = 0
                for start, end in predicted_segments:
                    segment_count += 1
                    segment_data = series[start:end+1]
                    segment_name = f"Series {sample_index} Segment {segment_count}"
                    segment_save_path = os.path.join(series_dir, f"{segment_count}.png")
                    
                    plot_single_segment(segment_data, segment_name, segment_save_path, color='#267BB6')
                    print(f"    保存分割段图片: {segment_save_path}")
                
                print(f"  共生成 {segment_count} 个分割段图片")
                success_count += 1
            else:
                print(f"  样本 {sample_index} Profile数据无效，无法生成图表")
                
        except Exception as e:
            print(f"  处理样本 {sample_index} 时出错: {e}")
    
    print(f"\n可视化完成！成功生成 {success_count} 个图表，保存在 {output_dir} 目录中")


def main():
    """主函数"""
    print("=" * 60)
    print("时间序列分割和覆盖率计算程序")
    print("=" * 60)
    
    try:
        # 处理数据集
        result_df = process_dataset()
        
        # 生成可视化图表
        visualize_sample_data()
        
        # 创建简化的结果CSV文件，只包含Index、Dataset、Predict_Positions、Size四列
        results_csv = create_results_csv(result_df)
        output_path = 'vis_output/results.csv'
        results_csv.to_csv(output_path, index=False)
        print(f"结果已保存到: {output_path}")
        print(f"CSV文件包含列: {list(results_csv.columns)}")
        
        # 计算总体统计信息
        print("\n" + "=" * 60)
        print("总体统计信息")
        print("=" * 60)
        
        # 统计有效处理的行数
        valid_rows = result_df[result_df['Coverage_Metrics'].notna()]
        print(f"总行数: {len(result_df)}")
        print(f"成功处理的行数: {len(valid_rows)}")
        
        if len(valid_rows) > 0:
            # 计算平均覆盖率指标
            all_metrics = []
            for metrics_str in valid_rows['Coverage_Metrics']:
                try:
                    metrics = ast.literal_eval(metrics_str)
                    all_metrics.append(metrics)
                except:
                    continue
            
            if all_metrics:
                avg_metrics = {}
                for key in all_metrics[0].keys():
                    # 只对数值类型的指标计算平均值
                    values = [m[key] for m in all_metrics if isinstance(m[key], (int, float))]
                    if values:
                        avg_metrics[key] = np.mean(values)
                    else:
                        # 对于非数值类型，计算统计信息
                        all_values = [m[key] for m in all_metrics]
                        avg_metrics[key] = f"共{len(all_values)}个值"
                
                print(f"\n平均覆盖率指标:")
                for metric, value in avg_metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"  {metric}: {value:.4f}")
                    else:
                        print(f"  {metric}: {value}")
        
        print("\n程序执行完成！")
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()