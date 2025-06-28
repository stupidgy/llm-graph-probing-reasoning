#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量GNN解释分析脚本
==================

这个脚本用于批量分析think和nothink数据的图神经网络解释结果，
生成解释文件并进行统计分析。
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict
import pandas as pd
import seaborn as sns

def run_explanation(model_path, think_path, nothink_path, layer, 
                   sample_idx, is_nothink, output_dir, 
                   explanation_method='gnnexplainer', 
                   explanation_type='model',
                   gpu_id=None):
    """运行单个样本的解释分析"""
    
    # 构建输出目录
    sample_type = 'nothink' if is_nothink else 'think'
    sample_output_dir = os.path.join(output_dir, f"layer_{layer}", sample_type, f"sample_{sample_idx}")
    os.makedirs(sample_output_dir, exist_ok=True)
    
    # 构建命令
    cmd = [
        'python', 'explain_model.py',
        '--model_path', model_path,
        '--think_path', think_path,
        '--nothink_path', nothink_path,
        '--llm_layer', str(layer),
        '--sample_idx', str(sample_idx),
        '--output_dir', sample_output_dir,
        '--explanation_method', explanation_method,
        '--explanation_type', explanation_type,
        '--fast_mode'
    ]
    
    # 添加GPU ID选择（优先使用空闲的GPU）
    if gpu_id is not None:
        cmd.extend(['--gpu_id', str(gpu_id)])
    
    if is_nothink:
        cmd.append('--load_nothink')
    
    print(f"正在分析 {sample_type} 样本 {sample_idx}...")
    print(f"命令: {' '.join(cmd)}")
    
    try:
        # 设置环境变量来限制GPU内存使用
        env = os.environ.copy()
        env['CUDA_LAUNCH_BLOCKING'] = '1'  # 启用同步错误报告
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'  # 限制内存分配
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='.', env=env)
        if result.returncode == 0:
            print(f"成功分析 {sample_type} 样本 {sample_idx}")
            return sample_output_dir
        else:
            print(f"分析 {sample_type} 样本 {sample_idx} 失败:")
            print(f"错误输出: {result.stderr}")
            return None
    except Exception as e:
        print(f"运行解释分析时出错: {e}")
        return None

def load_explanation_results(result_dir, explanation_method='gnnexplainer'):
    """加载解释结果"""
    edge_mask_path = os.path.join(result_dir, f"edge_mask_{explanation_method}.pt")
    node_mask_path = os.path.join(result_dir, f"node_mask_{explanation_method}.pt")
    
    results = {}
    
    if os.path.exists(edge_mask_path):
        results['edge_mask'] = torch.load(edge_mask_path, map_location='cpu')
    
    if os.path.exists(node_mask_path):
        results['node_mask'] = torch.load(node_mask_path, map_location='cpu')
    
    return results

def analyze_explanation_statistics(results_dict, output_dir):
    """分析解释结果的统计特性"""
    
    print("\n开始统计分析...")
    
    # 存储所有统计数据
    stats = {
        'think': {'node_importance': [], 'edge_importance': [], 'top_nodes': [], 'top_edges': []},
        'nothink': {'node_importance': [], 'edge_importance': [], 'top_nodes': [], 'top_edges': []}
    }
    
    # 分别处理think和nothink数据
    for sample_type in ['think', 'nothink']:
        if sample_type not in results_dict:
            continue
            
        print(f"\n分析 {sample_type} 数据...")
        
        for sample_idx, results in results_dict[sample_type].items():
            if not results:
                continue
                
            # 节点重要性分析
            if 'node_mask' in results:
                node_mask = results['node_mask']
                # 计算每个节点的总重要性
                if node_mask.dim() > 1:
                    node_importance = node_mask.sum(dim=1)
                else:
                    node_importance = node_mask
                
                stats[sample_type]['node_importance'].append(node_importance.numpy())
                
                # 获取前10个重要节点
                top_nodes = torch.argsort(node_importance, descending=True)[:10]
                stats[sample_type]['top_nodes'].append(top_nodes.numpy())
            
            # 边重要性分析
            if 'edge_mask' in results:
                edge_mask = results['edge_mask']
                stats[sample_type]['edge_importance'].append(edge_mask.numpy())
                
                # 获取前10个重要边
                top_edges = torch.argsort(edge_mask, descending=True)[:10]
                stats[sample_type]['top_edges'].append(top_edges.numpy())
    
    # 保存详细统计数据
    save_detailed_statistics(stats, output_dir)
    
    # 生成可视化对比
    generate_comparison_plots(stats, output_dir)
    
    # 生成统计报告
    generate_statistical_report(stats, output_dir)
    
    return stats

def save_detailed_statistics(stats, output_dir):
    """保存详细的统计数据"""
    
    stats_dir = os.path.join(output_dir, 'statistics')
    os.makedirs(stats_dir, exist_ok=True)
    
    for sample_type in ['think', 'nothink']:
        if not stats[sample_type]['node_importance']:
            continue
            
        # 保存节点重要性统计
        node_stats = {
            'sample_count': len(stats[sample_type]['node_importance']),
            'node_importance_means': [],
            'node_importance_stds': [],
            'top_nodes_frequency': defaultdict(int)
        }
        
        for node_imp in stats[sample_type]['node_importance']:
            node_stats['node_importance_means'].append(float(np.mean(node_imp)))
            node_stats['node_importance_stds'].append(float(np.std(node_imp)))
        
        # 统计最重要节点的频率
        for top_nodes in stats[sample_type]['top_nodes']:
            for node in top_nodes:
                node_stats['top_nodes_frequency'][int(node)] += 1
        
        # 转换为普通字典以便JSON序列化
        node_stats['top_nodes_frequency'] = dict(node_stats['top_nodes_frequency'])
        
        # 保存节点统计
        with open(os.path.join(stats_dir, f'{sample_type}_node_stats.json'), 'w') as f:
            json.dump(node_stats, f, indent=2)
        
        # 保存边重要性统计
        if stats[sample_type]['edge_importance']:
            edge_stats = {
                'sample_count': len(stats[sample_type]['edge_importance']),
                'edge_importance_means': [],
                'edge_importance_stds': [],
                'top_edges_frequency': defaultdict(int)
            }
            
            for edge_imp in stats[sample_type]['edge_importance']:
                edge_stats['edge_importance_means'].append(float(np.mean(edge_imp)))
                edge_stats['edge_importance_stds'].append(float(np.std(edge_imp)))
            
            # 统计最重要边的频率
            for top_edges in stats[sample_type]['top_edges']:
                for edge in top_edges:
                    edge_stats['top_edges_frequency'][int(edge)] += 1
            
            edge_stats['top_edges_frequency'] = dict(edge_stats['top_edges_frequency'])
            
            # 保存边统计
            with open(os.path.join(stats_dir, f'{sample_type}_edge_stats.json'), 'w') as f:
                json.dump(edge_stats, f, indent=2)

def generate_comparison_plots(stats, output_dir):
    """生成对比可视化图表"""
    
    plots_dir = os.path.join(output_dir, 'comparison_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 节点重要性分布对比
    if stats['think']['node_importance'] and stats['nothink']['node_importance']:
        plt.figure(figsize=(12, 8))
        
        # 计算平均节点重要性
        think_node_means = [np.mean(imp) for imp in stats['think']['node_importance']]
        nothink_node_means = [np.mean(imp) for imp in stats['nothink']['node_importance']]
        
        plt.subplot(2, 2, 1)
        plt.hist(think_node_means, bins=20, alpha=0.7, label='Think', color='blue')
        plt.hist(nothink_node_means, bins=20, alpha=0.7, label='NoThink', color='red')
        plt.xlabel('Average Node Importance')
        plt.ylabel('Frequency')
        plt.title('Distribution of Average Node Importance')
        plt.legend()
        
        # 2. 节点重要性标准差对比
        think_node_stds = [np.std(imp) for imp in stats['think']['node_importance']]
        nothink_node_stds = [np.std(imp) for imp in stats['nothink']['node_importance']]
        
        plt.subplot(2, 2, 2)
        plt.hist(think_node_stds, bins=20, alpha=0.7, label='Think', color='blue')
        plt.hist(nothink_node_stds, bins=20, alpha=0.7, label='NoThink', color='red')
        plt.xlabel('Node Importance Std')
        plt.ylabel('Frequency')
        plt.title('Distribution of Node Importance Variability')
        plt.legend()
        
        # 3. 边重要性分布对比（如果有边数据）
        if stats['think']['edge_importance'] and stats['nothink']['edge_importance']:
            think_edge_means = [np.mean(imp) for imp in stats['think']['edge_importance']]
            nothink_edge_means = [np.mean(imp) for imp in stats['nothink']['edge_importance']]
            
            plt.subplot(2, 2, 3)
            plt.hist(think_edge_means, bins=20, alpha=0.7, label='Think', color='blue')
            plt.hist(nothink_edge_means, bins=20, alpha=0.7, label='NoThink', color='red')
            plt.xlabel('Average Edge Importance')
            plt.ylabel('Frequency')
            plt.title('Distribution of Average Edge Importance')
            plt.legend()
            
            # 4. 边重要性标准差对比
            think_edge_stds = [np.std(imp) for imp in stats['think']['edge_importance']]
            nothink_edge_stds = [np.std(imp) for imp in stats['nothink']['edge_importance']]
            
            plt.subplot(2, 2, 4)
            plt.hist(think_edge_stds, bins=20, alpha=0.7, label='Think', color='blue')
            plt.hist(nothink_edge_stds, bins=20, alpha=0.7, label='NoThink', color='red')
            plt.xlabel('Edge Importance Std')
            plt.ylabel('Frequency')
            plt.title('Distribution of Edge Importance Variability')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'importance_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. 最重要节点频率对比
    generate_top_elements_comparison(stats, plots_dir, 'nodes')
    
    # 6. 最重要边频率对比（如果有边数据）
    if stats['think']['edge_importance'] and stats['nothink']['edge_importance']:
        generate_top_elements_comparison(stats, plots_dir, 'edges')

def generate_top_elements_comparison(stats, plots_dir, element_type):
    """生成最重要元素的频率对比图"""
    
    # 统计频率
    think_freq = defaultdict(int)
    nothink_freq = defaultdict(int)
    
    key = f'top_{element_type}'
    
    for top_elements in stats['think'][key]:
        for element in top_elements:
            think_freq[int(element)] += 1
    
    for top_elements in stats['nothink'][key]:
        for element in top_elements:
            nothink_freq[int(element)] += 1
    
    # 获取最频繁的元素
    all_elements = set(list(think_freq.keys()) + list(nothink_freq.keys()))
    top_elements = sorted(all_elements, key=lambda x: think_freq[x] + nothink_freq[x], reverse=True)[:20]
    
    if not top_elements:
        return
    
    think_counts = [think_freq[elem] for elem in top_elements]
    nothink_counts = [nothink_freq[elem] for elem in top_elements]
    
    plt.figure(figsize=(12, 8))
    x = np.arange(len(top_elements))
    width = 0.35
    
    plt.bar(x - width/2, think_counts, width, label='Think', color='blue', alpha=0.7)
    plt.bar(x + width/2, nothink_counts, width, label='NoThink', color='red', alpha=0.7)
    
    plt.xlabel(f'{element_type.capitalize()} Index')
    plt.ylabel('Frequency in Top 10')
    plt.title(f'Most Frequently Important {element_type.capitalize()}')
    plt.xticks(x, [str(elem) for elem in top_elements], rotation=45)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(plots_dir, f'top_{element_type}_frequency.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_statistical_report(stats, output_dir):
    """生成统计分析报告"""
    
    report_path = os.path.join(output_dir, 'statistical_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("GNN解释结果统计分析报告\n")
        f.write("=" * 50 + "\n\n")
        
        for sample_type in ['think', 'nothink']:
            if not stats[sample_type]['node_importance']:
                continue
                
            f.write(f"{sample_type.upper()} 数据分析结果:\n")
            f.write("-" * 30 + "\n")
            
            # 节点重要性统计
            node_importances = stats[sample_type]['node_importance']
            f.write(f"样本数量: {len(node_importances)}\n")
            
            if node_importances:
                # 计算整体统计
                all_node_values = np.concatenate(node_importances)
                f.write(f"节点重要性统计:\n")
                f.write(f"  总节点数: {len(all_node_values)}\n")
                f.write(f"  平均重要性: {np.mean(all_node_values):.6f}\n")
                f.write(f"  标准差: {np.std(all_node_values):.6f}\n")
                f.write(f"  最大重要性: {np.max(all_node_values):.6f}\n")
                f.write(f"  最小重要性: {np.min(all_node_values):.6f}\n")
                
                # 每个样本的平均重要性统计
                sample_means = [np.mean(imp) for imp in node_importances]
                f.write(f"样本间平均重要性统计:\n")
                f.write(f"  平均值: {np.mean(sample_means):.6f}\n")
                f.write(f"  标准差: {np.std(sample_means):.6f}\n")
            
            # 边重要性统计
            edge_importances = stats[sample_type]['edge_importance']
            if edge_importances:
                all_edge_values = np.concatenate(edge_importances)
                f.write(f"边重要性统计:\n")
                f.write(f"  总边数: {len(all_edge_values)}\n")
                f.write(f"  平均重要性: {np.mean(all_edge_values):.6f}\n")
                f.write(f"  标准差: {np.std(all_edge_values):.6f}\n")
                f.write(f"  最大重要性: {np.max(all_edge_values):.6f}\n")
                f.write(f"  最小重要性: {np.min(all_edge_values):.6f}\n")
            
            # 最频繁的重要节点
            top_nodes_freq = defaultdict(int)
            for top_nodes in stats[sample_type]['top_nodes']:
                for node in top_nodes:
                    top_nodes_freq[int(node)] += 1
            
            if top_nodes_freq:
                sorted_nodes = sorted(top_nodes_freq.items(), key=lambda x: x[1], reverse=True)
                f.write(f"最频繁的重要节点 (前10):\n")
                for i, (node, freq) in enumerate(sorted_nodes[:10]):
                    f.write(f"  {i+1}. 节点 {node}: 出现 {freq} 次\n")
            
            f.write("\n")
        
        # 对比分析
        if stats['think']['node_importance'] and stats['nothink']['node_importance']:
            f.write("THINK vs NOTHINK 对比分析:\n")
            f.write("-" * 30 + "\n")
            
            think_means = [np.mean(imp) for imp in stats['think']['node_importance']]
            nothink_means = [np.mean(imp) for imp in stats['nothink']['node_importance']]
            
            f.write(f"节点重要性平均值对比:\n")
            f.write(f"  Think 平均: {np.mean(think_means):.6f} ± {np.std(think_means):.6f}\n")
            f.write(f"  NoThink 平均: {np.mean(nothink_means):.6f} ± {np.std(nothink_means):.6f}\n")
            
            # 简单的统计检验（t检验）
            from scipy import stats as scipy_stats
            try:
                t_stat, p_value = scipy_stats.ttest_ind(think_means, nothink_means)
                f.write(f"  t检验结果: t={t_stat:.4f}, p={p_value:.6f}\n")
                if p_value < 0.05:
                    f.write(f"  结论: 两组间存在显著差异 (p < 0.05)\n")
                else:
                    f.write(f"  结论: 两组间无显著差异 (p >= 0.05)\n")
            except Exception as e:
                f.write(f"  t检验失败: {e}\n")
    
    print(f"统计报告已保存到: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='批量GNN解释分析')
    parser.add_argument('--model_path', type=str, 
                        default='saves/binary_classification/layer_14/best_model_density-1.0_dim-32_hop-1.pth',
                        help='训练好的模型路径')
    parser.add_argument('--think_path', type=str, 
                        default='data/graph_probing/data4/huguangyi/models/Qwen/Qwen3-0.6B_no_thinking',
                        help='think数据集路径')
    parser.add_argument('--nothink_path', type=str,
                        default='data/graph_probing/data4/huguangyi/models/Qwen/Qwen3-0.6B_nothink',
                        help='nothink数据集路径')
    parser.add_argument('--layer', type=int, default=14,
                        help='要分析的LLM层数')
    parser.add_argument('--num_samples', type=int, default=1855,
                        help='每种类型要分析的样本数量')
    parser.add_argument('--output_dir', type=str, default='explanation_results',
                        help='输出目录')
    parser.add_argument('--explanation_method', type=str, default='gnnexplainer',
                        choices=['gnnexplainer', 'captum_ig', 'captum_saliency'],
                        help='解释方法')
    parser.add_argument('--explanation_type', type=str, default='model',
                        choices=['model', 'phenomenon'],
                        help='解释类型')
    parser.add_argument('--skip_explanation', action='store_true',
                        help='跳过解释生成，直接进行统计分析（假设解释文件已存在）')
    parser.add_argument('--only_nothink', action='store_true',
                        help='只重新生成nothink样本的解释，保留现有的think样本解释')
    parser.add_argument('--only_think', action='store_true',
                        help='只重新生成think样本的解释，保留现有的nothink样本解释')
    parser.add_argument('--force_regenerate', action='store_true',
                        help='强制重新生成解释，即使文件已存在')
    parser.add_argument('--gpu_id', type=int, default=None,
                        help='指定使用的GPU ID，如果不指定则自动选择最空闲的GPU')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='批处理大小，减少此值可以降低内存使用')
    
    args = parser.parse_args()
    
    # 选择合适的GPU
    if args.gpu_id is None:
        print("自动选择最空闲的GPU...")
        try:
            import subprocess
            import re
            
            # 获取GPU内存使用情况
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            gpu_info = []
            for line in result.stdout.strip().split('\n'):
                parts = line.split(', ')
                gpu_id = int(parts[0])
                used_mem = int(parts[1])
                total_mem = int(parts[2])
                free_mem = total_mem - used_mem
                gpu_info.append((gpu_id, free_mem, used_mem, total_mem))
            
            # 选择空闲内存最多的GPU
            gpu_info.sort(key=lambda x: x[1], reverse=True)
            selected_gpu = gpu_info[0][0]
            args.gpu_id = selected_gpu
            
            print(f"选择的GPU: {selected_gpu} (空闲内存: {gpu_info[0][1]}MB / {gpu_info[0][3]}MB)")
            
            # 如果最空闲的GPU内存也不足（少于10GB），给出警告
            if gpu_info[0][1] < 10240:  # 10GB
                print(f"警告: 选择的GPU内存不足，仅有 {gpu_info[0][1]/1024:.1f}GB 空闲内存")
                print("建议减少batch_size或使用其他GPU")
                
        except Exception as e:
            print(f"无法获取GPU信息: {e}")
            print("使用默认GPU 0")
            args.gpu_id = 0
    else:
        print(f"使用指定的GPU: {args.gpu_id}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 存储所有结果
    all_results = {'think': {}, 'nothink': {}}
    
    if not args.skip_explanation:
        print("开始批量生成解释...")
        
        # 确定要处理的样本类型
        process_think = not args.only_nothink
        process_nothink = not args.only_think
        
        # 分批处理以减少内存压力
        batch_size = args.batch_size
        num_batches = (args.num_samples + batch_size - 1) // batch_size
        
        # 分析think样本
        if process_think:
            print(f"\n分析think样本（分{num_batches}批，每批{batch_size}个）...")
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, args.num_samples)
                print(f"处理第 {batch_idx + 1}/{num_batches} 批 think 样本 ({start_idx}-{end_idx-1})...")
                
                for i in range(start_idx, end_idx):
                    # 检查文件是否已存在，除非强制重新生成
                    result_dir = os.path.join(args.output_dir, f"layer_{args.layer}", "think", f"sample_{i}")
                    
                    should_generate = args.force_regenerate or not os.path.exists(
                        os.path.join(result_dir, f"edge_mask_{args.explanation_method}.pt")
                    )
                    
                    if should_generate:
                        result_dir = run_explanation(
                            args.model_path, args.think_path, args.nothink_path, 
                            args.layer, i, False, args.output_dir,
                            args.explanation_method, args.explanation_type,
                            gpu_id=args.gpu_id
                        )
                    else:
                        print(f"跳过think样本 {i}（文件已存在）")
                    
                    if result_dir and os.path.exists(result_dir):
                        results = load_explanation_results(result_dir, args.explanation_method)
                        all_results['think'][i] = results
                
                # 每批之间稍作停顿，释放GPU内存
                import time
                time.sleep(2)
        else:
            print("跳过think样本生成，从现有文件加载...")
            # 从现有文件加载think结果
            for i in range(args.num_samples):
                result_dir = os.path.join(args.output_dir, f"layer_{args.layer}", "think", f"sample_{i}")
                if os.path.exists(result_dir):
                    results = load_explanation_results(result_dir, args.explanation_method)
                    if results:
                        all_results['think'][i] = results
        
        # 分析nothink样本
        if process_nothink:
            print(f"\n分析nothink样本（分{num_batches}批，每批{batch_size}个）...")
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, args.num_samples)
                print(f"处理第 {batch_idx + 1}/{num_batches} 批 nothink 样本 ({start_idx}-{end_idx-1})...")
                
                for i in range(start_idx, end_idx):
                    # 检查文件是否已存在，除非强制重新生成
                    result_dir = os.path.join(args.output_dir, f"layer_{args.layer}", "nothink", f"sample_{i}")
                    
                    should_generate = args.force_regenerate or not os.path.exists(
                        os.path.join(result_dir, f"edge_mask_{args.explanation_method}.pt")
                    )
                    
                    if should_generate:
                        result_dir = run_explanation(
                            args.model_path, args.think_path, args.nothink_path, 
                            args.layer, i, True, args.output_dir,
                            args.explanation_method, args.explanation_type,
                            gpu_id=args.gpu_id
                        )
                    else:
                        print(f"跳过nothink样本 {i}（文件已存在）")
                    
                    if result_dir and os.path.exists(result_dir):
                        results = load_explanation_results(result_dir, args.explanation_method)
                        all_results['nothink'][i] = results
                
                # 每批之间稍作停顿，释放GPU内存
                import time
                time.sleep(2)
        else:
            print("跳过nothink样本生成，从现有文件加载...")
            # 从现有文件加载nothink结果
            for i in range(args.num_samples):
                result_dir = os.path.join(args.output_dir, f"layer_{args.layer}", "nothink", f"sample_{i}")
                if os.path.exists(result_dir):
                    results = load_explanation_results(result_dir, args.explanation_method)
                    if results:
                        all_results['nothink'][i] = results
    else:
        print("跳过解释生成，从现有文件加载结果...")
        
        # 从现有文件加载结果
        for sample_type in ['think', 'nothink']:
            for i in range(args.num_samples):
                result_dir = os.path.join(args.output_dir, f"layer_{args.layer}", sample_type, f"sample_{i}")
                if os.path.exists(result_dir):
                    results = load_explanation_results(result_dir, args.explanation_method)
                    if results:
                        all_results[sample_type][i] = results
    
    # 统计分析
    print(f"\n加载的结果总数: Think={len(all_results['think'])}, NoThink={len(all_results['nothink'])}")
    
    if all_results['think'] or all_results['nothink']:
        analyze_explanation_statistics(all_results, args.output_dir)
        print(f"\n分析完成！结果保存在: {args.output_dir}")
    else:
        print("未找到有效的解释结果，请检查路径和参数。")

if __name__ == "__main__":
    main() 