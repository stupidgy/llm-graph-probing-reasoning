#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重要性分析可视化脚本
==================

分析explanation中layer的节点和边重要性，计算类内平均和全局平均，并进行可视化对比分析。
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy import stats
import argparse
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

# 设置matplotlib中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False

class ImportanceAnalyzer:
    def __init__(self, explanation_results_dir, num_nodes=1024):
        """
        初始化重要性分析器
        
        Args:
            explanation_results_dir: explanation结果目录路径
            num_nodes: 全连接网络的节点数，默认1024
        """
        self.results_dir = explanation_results_dir
        self.num_nodes = num_nodes  # 添加节点数参数
        self.node_data = {'think': [], 'nothink': []}
        self.edge_data = {'think': [], 'nothink': []}
        self.avg_matrices = {}  # 存储平均矩阵
    
    def get_edge_endpoints_from_index(self, edge_idx):
        """
        从边索引计算端点（适用于全连接网络）
        
        Args:
            edge_idx: 边的索引
            
        Returns:
            tuple: (src_node, dst_node)
        """
        src = edge_idx // self.num_nodes
        dst = edge_idx % self.num_nodes
        return src, dst
        
    def load_explanation_data(self, layer_idx=None):
        """
        加载explanation数据
        
        Args:
            layer_idx: 指定layer，如果为None则加载所有layer
        """
        print("正在加载explanation数据...")
        
        # 遍历layer目录
        layer_dirs = [d for d in os.listdir(self.results_dir) if d.startswith('layer_')]
        if layer_idx is not None:
            layer_dirs = [d for d in layer_dirs if d == f'layer_{layer_idx}']
        
        for layer_dir in layer_dirs:
            layer_path = os.path.join(self.results_dir, layer_dir)
            print(f"处理 {layer_dir}...")
            
            # 处理think和nothink数据
            for category in ['think', 'nothink']:
                category_path = os.path.join(layer_path, category)
                if not os.path.exists(category_path):
                    continue
                
                # 遍历样本目录
                sample_dirs = [d for d in os.listdir(category_path) if d.startswith('sample_')]
                for sample_dir in sample_dirs:
                    sample_path = os.path.join(category_path, sample_dir)
                    
                    # 加载节点和边掩码
                    node_mask_path = os.path.join(sample_path, "node_mask_gnnexplainer.pt")
                    edge_mask_path = os.path.join(sample_path, "edge_mask_gnnexplainer.pt")
                    
                    sample_data = {
                        'layer': int(layer_dir.split('_')[1]),
                        'file_path': sample_path  # 添加文件路径信息
                    }
                    
                    if os.path.exists(node_mask_path):
                        node_mask = torch.load(node_mask_path, map_location='cpu')
                        sample_data['node_importance'] = node_mask.numpy()
                        
                    if os.path.exists(edge_mask_path):
                        edge_mask = torch.load(edge_mask_path, map_location='cpu')
                        sample_data['edge_importance'] = edge_mask.numpy()
                    
                    if 'node_importance' in sample_data or 'edge_importance' in sample_data:
                        if category == 'think':
                            self.node_data['think'].append(sample_data)
                        else:
                            self.node_data['nothink'].append(sample_data)
        
        print(f"加载完成: think样本 {len(self.node_data['think'])} 个, nothink样本 {len(self.node_data['nothink'])} 个")
    
    def compute_class_and_global_averages(self):
        """
        计算类内平均和全局平均
        """
        print("计算类内平均和全局平均...")
        
        results = {
            'node_analysis': {
                'think': {'class_avg': [], 'global_avg': []},
                'nothink': {'class_avg': [], 'global_avg': []}
            },
            'edge_analysis': {
                'think': {'class_avg': [], 'global_avg': []}, 
                'nothink': {'class_avg': [], 'global_avg': []}
            }
        }
        
        # 分析节点重要性
        for category in ['think', 'nothink']:
            all_node_importance = []
            class_averages = []
            
            for sample in self.node_data[category]:
                if 'node_importance' in sample:
                    node_imp = sample['node_importance']
                    if node_imp.ndim > 1:
                        # 如果是多维，沿axis=1求和（假设每行是一个节点的多个特征）
                        node_imp = np.sum(node_imp, axis=1)
                    
                    # 类内平均（每个样本内节点的平均重要性）
                    class_avg = np.mean(node_imp)
                    class_averages.append(class_avg)
                    
                    # 收集所有节点重要性用于全局平均
                    all_node_importance.extend(node_imp.flatten())
            
            # 全局平均（所有节点重要性的平均）
            global_avg = np.mean(all_node_importance) if all_node_importance else 0
            
            results['node_analysis'][category]['class_avg'] = class_averages
            results['node_analysis'][category]['global_avg'] = global_avg
        
        # 分析边重要性
        for category in ['think', 'nothink']:
            all_edge_importance = []
            class_averages = []
            
            for sample in self.node_data[category]:
                if 'edge_importance' in sample:
                    edge_imp = sample['edge_importance']
                    
                    # 类内平均（每个样本内边的平均重要性）
                    class_avg = np.mean(edge_imp)
                    class_averages.append(class_avg)
                    
                    # 收集所有边重要性用于全局平均
                    all_edge_importance.extend(edge_imp.flatten())
            
            # 全局平均（所有边重要性的平均）
            global_avg = np.mean(all_edge_importance) if all_edge_importance else 0
            
            results['edge_analysis'][category]['class_avg'] = class_averages
            results['edge_analysis'][category]['global_avg'] = global_avg
        
        return results
    
    def compute_average_matrices(self):
        """
        计算think和nothink的平均节点和边重要性矩阵
        """
        print("计算平均重要性矩阵...")
        
        avg_matrices = {
            'think': {'node_importance': None, 'edge_importance': None},
            'nothink': {'node_importance': None, 'edge_importance': None},
            'combined': {'node_importance': None, 'edge_importance': None}  # 添加合并均值
        }
        
        # 收集所有数据用于计算合并均值
        all_combined_node_importance = []
        all_combined_edge_importance = []
        
        for category in ['think', 'nothink']:
            all_node_importance = []
            all_edge_importance = []
            
            for sample in self.node_data[category]:
                if 'node_importance' in sample:
                    node_imp = sample['node_importance']
                    if node_imp.ndim > 1:
                        # 如果是多维，沿axis=1求和
                        node_imp = np.sum(node_imp, axis=1)
                    all_node_importance.append(node_imp)
                    all_combined_node_importance.append(node_imp)  # 添加到合并列表
                
                if 'edge_importance' in sample:
                    edge_imp = sample['edge_importance']
                    all_edge_importance.append(edge_imp)
                    all_combined_edge_importance.append(edge_imp)  # 添加到合并列表
            
            # 计算类别平均节点重要性
            if all_node_importance:
                # 确保所有样本的节点数相同
                min_nodes = min(len(imp) for imp in all_node_importance)
                truncated_node_imp = [imp[:min_nodes] for imp in all_node_importance]
                avg_matrices[category]['node_importance'] = np.mean(truncated_node_imp, axis=0)
                print(f"{category} 平均节点重要性形状: {avg_matrices[category]['node_importance'].shape}")
            
            # 计算类别平均边重要性
            if all_edge_importance:
                # 确保所有样本的边数相同
                min_edges = min(len(imp) for imp in all_edge_importance)
                truncated_edge_imp = [imp[:min_edges] for imp in all_edge_importance]
                avg_matrices[category]['edge_importance'] = np.mean(truncated_edge_imp, axis=0)
                print(f"{category} 平均边重要性形状: {avg_matrices[category]['edge_importance'].shape}")
        
        # 计算合并的平均重要性
        if all_combined_node_importance:
            min_nodes_combined = min(len(imp) for imp in all_combined_node_importance)
            truncated_combined_node = [imp[:min_nodes_combined] for imp in all_combined_node_importance]
            avg_matrices['combined']['node_importance'] = np.mean(truncated_combined_node, axis=0)
            print(f"合并 平均节点重要性形状: {avg_matrices['combined']['node_importance'].shape}")
        
        if all_combined_edge_importance:
            min_edges_combined = min(len(imp) for imp in all_combined_edge_importance)
            truncated_combined_edge = [imp[:min_edges_combined] for imp in all_combined_edge_importance]
            avg_matrices['combined']['edge_importance'] = np.mean(truncated_combined_edge, axis=0)
            print(f"合并 平均边重要性形状: {avg_matrices['combined']['edge_importance'].shape}")
        
        self.avg_matrices = avg_matrices
        return avg_matrices
    
    def save_average_matrices(self, output_dir):
        """
        保存平均重要性矩阵
        """
        print("保存平均重要性矩阵...")
        os.makedirs(output_dir, exist_ok=True)
        
        for category in ['think', 'nothink', 'combined']:  # 添加combined类别
            if self.avg_matrices[category]['node_importance'] is not None:
                np.save(
                    os.path.join(output_dir, f'{category}_avg_node_importance.npy'),
                    self.avg_matrices[category]['node_importance']
                )
                
            if self.avg_matrices[category]['edge_importance'] is not None:
                np.save(
                    os.path.join(output_dir, f'{category}_avg_edge_importance.npy'),
                    self.avg_matrices[category]['edge_importance']
                )
        
        print("平均重要性矩阵已保存")
    
    def create_comparison_visualizations(self, analysis_results, output_dir):
        """
        创建对比可视化图表
        """
        print("创建可视化图表...")
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置图表风格 - 使用更兼容的方式
        try:
            import seaborn as sns
            sns.set_style("whitegrid")
        except ImportError:
            # 如果没有seaborn，使用matplotlib默认样式
            plt.style.use('default')
        
        # 设置matplotlib参数
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        
        colors = {'think': '#2E8B57', 'nothink': '#DC143C'}
        
        # 1. 节点重要性分析
        self._plot_node_importance_analysis(analysis_results, output_dir, colors)
        
        # 2. 边重要性分析
        self._plot_edge_importance_analysis(analysis_results, output_dir, colors)
        
        # 3. 综合对比分析
        self._plot_comprehensive_comparison(analysis_results, output_dir, colors)
        
    def _plot_node_importance_analysis(self, results, output_dir, colors):
        """节点重要性分析图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Node Importance Analysis', fontsize=16, fontweight='bold')
        
        # 1. 类内平均分布对比
        ax1 = axes[0, 0]
        think_class_avg = results['node_analysis']['think']['class_avg']
        nothink_class_avg = results['node_analysis']['nothink']['class_avg']
        
        if think_class_avg and nothink_class_avg:
            ax1.hist(think_class_avg, bins=30, alpha=0.7, label='Think', 
                    color=colors['think'], density=True)
            ax1.hist(nothink_class_avg, bins=30, alpha=0.7, label='NoThink', 
                    color=colors['nothink'], density=True)
            ax1.set_xlabel('Class-wise Average Node Importance')
            ax1.set_ylabel('Density')
            ax1.set_title('Class-wise Average Node Importance Distribution')
            ax1.legend()
        
        # 2. 类内平均箱线图
        ax2 = axes[0, 1]
        if think_class_avg and nothink_class_avg:
            data = [think_class_avg, nothink_class_avg]
            labels = ['Think', 'NoThink']
            bp = ax2.boxplot(data, labels=labels, patch_artist=True)
            bp['boxes'][0].set_facecolor(colors['think'])
            bp['boxes'][1].set_facecolor(colors['nothink'])
            ax2.set_ylabel('Class-wise Average Node Importance')
            ax2.set_title('Class-wise Average Node Importance Boxplot')
        
        # 3. 全局平均对比
        ax3 = axes[1, 0]
        think_global = results['node_analysis']['think']['global_avg']
        nothink_global = results['node_analysis']['nothink']['global_avg']
        
        categories = ['Think', 'NoThink']
        values = [think_global, nothink_global]
        bars = ax3.bar(categories, values, color=[colors['think'], colors['nothink']], alpha=0.7)
        ax3.set_ylabel('Global Average Node Importance')
        ax3.set_title('Global Average Node Importance Comparison')
        
        # 在柱子上添加数值
        for bar, value in zip(bars, values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.4f}', ha='center', va='bottom')
        
        # 4. 统计显著性检验
        ax4 = axes[1, 1]
        if think_class_avg and nothink_class_avg:
            # 进行t检验
            t_stat, p_value = stats.ttest_ind(think_class_avg, nothink_class_avg)
            
            # 创建统计信息文本
            stats_text = f'Statistical Test Results:\n'
            stats_text += f't-statistic: {t_stat:.4f}\n'
            stats_text += f'p-value: {p_value:.6f}\n'
            stats_text += f'Significant: {"Yes" if p_value < 0.05 else "No"} (α=0.05)'
            
            ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=12,
                    verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax4.set_title('Statistical Significance Test')
            ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'node_importance_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_edge_importance_analysis(self, results, output_dir, colors):
        """边重要性分析图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Edge Importance Analysis', fontsize=16, fontweight='bold')
        
        # 1. 类内平均分布对比
        ax1 = axes[0, 0]
        think_class_avg = results['edge_analysis']['think']['class_avg']
        nothink_class_avg = results['edge_analysis']['nothink']['class_avg']
        
        if think_class_avg and nothink_class_avg:
            ax1.hist(think_class_avg, bins=30, alpha=0.7, label='Think', 
                    color=colors['think'], density=True)
            ax1.hist(nothink_class_avg, bins=30, alpha=0.7, label='NoThink', 
                    color=colors['nothink'], density=True)
            ax1.set_xlabel('Class-wise Average Edge Importance')
            ax1.set_ylabel('Density')
            ax1.set_title('Class-wise Average Edge Importance Distribution')
            ax1.legend()
        
        # 2. 类内平均箱线图
        ax2 = axes[0, 1]
        if think_class_avg and nothink_class_avg:
            data = [think_class_avg, nothink_class_avg]
            labels = ['Think', 'NoThink']
            bp = ax2.boxplot(data, labels=labels, patch_artist=True)
            bp['boxes'][0].set_facecolor(colors['think'])
            bp['boxes'][1].set_facecolor(colors['nothink'])
            ax2.set_ylabel('Class-wise Average Edge Importance')
            ax2.set_title('Class-wise Average Edge Importance Boxplot')
        
        # 3. 全局平均对比
        ax3 = axes[1, 0]
        think_global = results['edge_analysis']['think']['global_avg']
        nothink_global = results['edge_analysis']['nothink']['global_avg']
        
        categories = ['Think', 'NoThink']
        values = [think_global, nothink_global]
        bars = ax3.bar(categories, values, color=[colors['think'], colors['nothink']], alpha=0.7)
        ax3.set_ylabel('Global Average Edge Importance')
        ax3.set_title('Global Average Edge Importance Comparison')
        
        # 在柱子上添加数值
        for bar, value in zip(bars, values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.4f}', ha='center', va='bottom')
        
        # 4. 统计显著性检验
        ax4 = axes[1, 1]
        if think_class_avg and nothink_class_avg:
            # 进行t检验
            t_stat, p_value = stats.ttest_ind(think_class_avg, nothink_class_avg)
            
            # 创建统计信息文本
            stats_text = f'Statistical Test Results:\n'
            stats_text += f't-statistic: {t_stat:.4f}\n'
            stats_text += f'p-value: {p_value:.6f}\n'
            stats_text += f'Significant: {"Yes" if p_value < 0.05 else "No"} (α=0.05)'
            
            ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=12,
                    verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax4.set_title('Statistical Significance Test')
            ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'edge_importance_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_comprehensive_comparison(self, results, output_dir, colors):
        """综合对比分析图表"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Node vs Edge Importance Comprehensive Comparison', fontsize=16, fontweight='bold')
        
        # 1. 类内平均对比 - 节点 vs 边
        ax1 = axes[0, 0]
        think_node_avg = np.mean(results['node_analysis']['think']['class_avg']) if results['node_analysis']['think']['class_avg'] else 0
        think_edge_avg = np.mean(results['edge_analysis']['think']['class_avg']) if results['edge_analysis']['think']['class_avg'] else 0
        nothink_node_avg = np.mean(results['node_analysis']['nothink']['class_avg']) if results['node_analysis']['nothink']['class_avg'] else 0
        nothink_edge_avg = np.mean(results['edge_analysis']['nothink']['class_avg']) if results['edge_analysis']['nothink']['class_avg'] else 0
        
        x = np.arange(2)
        width = 0.35
        
        think_values = [think_node_avg, think_edge_avg]
        nothink_values = [nothink_node_avg, nothink_edge_avg]
        
        ax1.bar(x - width/2, think_values, width, label='Think', color=colors['think'], alpha=0.7)
        ax1.bar(x + width/2, nothink_values, width, label='NoThink', color=colors['nothink'], alpha=0.7)
        
        ax1.set_xlabel('Importance Type')
        ax1.set_ylabel('Average Importance Value')
        ax1.set_title('Class-wise Average Importance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['Node', 'Edge'])
        ax1.legend()
        
        # 2. 全局平均对比
        ax2 = axes[0, 1]
        think_global_values = [results['node_analysis']['think']['global_avg'], 
                              results['edge_analysis']['think']['global_avg']]
        nothink_global_values = [results['node_analysis']['nothink']['global_avg'], 
                                results['edge_analysis']['nothink']['global_avg']]
        
        ax2.bar(x - width/2, think_global_values, width, label='Think', color=colors['think'], alpha=0.7)
        ax2.bar(x + width/2, nothink_global_values, width, label='NoThink', color=colors['nothink'], alpha=0.7)
        
        ax2.set_xlabel('Importance Type')
        ax2.set_ylabel('Global Average Importance Value')
        ax2.set_title('Global Average Importance Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['Node', 'Edge'])
        ax2.legend()
        
        # 3. 散点图 - 节点 vs 边重要性相关性
        ax3 = axes[0, 2]
        if (results['node_analysis']['think']['class_avg'] and 
            results['edge_analysis']['think']['class_avg']):
            
            think_node_ca = results['node_analysis']['think']['class_avg']
            think_edge_ca = results['edge_analysis']['think']['class_avg']
            nothink_node_ca = results['node_analysis']['nothink']['class_avg']
            nothink_edge_ca = results['edge_analysis']['nothink']['class_avg']
            
            # 确保两个列表长度相同
            min_len_think = min(len(think_node_ca), len(think_edge_ca))
            min_len_nothink = min(len(nothink_node_ca), len(nothink_edge_ca))
            
            if min_len_think > 0:
                ax3.scatter(think_node_ca[:min_len_think], think_edge_ca[:min_len_think], 
                           alpha=0.6, color=colors['think'], label='Think', s=30)
            
            if min_len_nothink > 0:
                ax3.scatter(nothink_node_ca[:min_len_nothink], nothink_edge_ca[:min_len_nothink], 
                           alpha=0.6, color=colors['nothink'], label='NoThink', s=30)
            
            ax3.set_xlabel('Node Importance (Class-wise Average)')
            ax3.set_ylabel('Edge Importance (Class-wise Average)')
            ax3.set_title('Node vs Edge Importance Correlation')
            ax3.legend()
        
        # 4. 小提琴图 - 节点重要性分布
        ax4 = axes[1, 0]
        if (results['node_analysis']['think']['class_avg'] and 
            results['node_analysis']['nothink']['class_avg']):
            
            data = [results['node_analysis']['think']['class_avg'],
                   results['node_analysis']['nothink']['class_avg']]
            labels = ['Think', 'NoThink']
            
            parts = ax4.violinplot(data, showmeans=True, showmedians=True)
            
            # 设置颜色
            for i, pc in enumerate(parts['bodies']):
                color = colors['think'] if i == 0 else colors['nothink']
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            
            ax4.set_xticks([1, 2])
            ax4.set_xticklabels(labels)
            ax4.set_ylabel('Node Importance (Class-wise Average)')
            ax4.set_title('Node Importance Distribution (Violin Plot)')
        
        # 5. 小提琴图 - 边重要性分布
        ax5 = axes[1, 1]
        if (results['edge_analysis']['think']['class_avg'] and 
            results['edge_analysis']['nothink']['class_avg']):
            
            data = [results['edge_analysis']['think']['class_avg'],
                   results['edge_analysis']['nothink']['class_avg']]
            labels = ['Think', 'NoThink']
            
            parts = ax5.violinplot(data, showmeans=True, showmedians=True)
            
            # 设置颜色
            for i, pc in enumerate(parts['bodies']):
                color = colors['think'] if i == 0 else colors['nothink']
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            
            ax5.set_xticks([1, 2])
            ax5.set_xticklabels(labels)
            ax5.set_ylabel('Edge Importance (Class-wise Average)')
            ax5.set_title('Edge Importance Distribution (Violin Plot)')
        
        # 6. 综合统计摘要
        ax6 = axes[1, 2]
        summary_text = self._generate_summary_statistics(results)
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax6.set_title('Statistical Summary')
        ax6.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comprehensive_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_summary_statistics(self, results):
        """生成统计摘要"""
        summary = "Importance Analysis Summary\n" + "="*25 + "\n\n"
        
        # 节点重要性统计
        summary += "Node Importance:\n"
        think_node_avg = results['node_analysis']['think']['class_avg']
        nothink_node_avg = results['node_analysis']['nothink']['class_avg']
        
        if think_node_avg and nothink_node_avg:
            summary += f"Think Class Avg: {np.mean(think_node_avg):.4f}±{np.std(think_node_avg):.4f}\n"
            summary += f"NoThink Class Avg: {np.mean(nothink_node_avg):.4f}±{np.std(nothink_node_avg):.4f}\n"
            summary += f"Think Global Avg: {results['node_analysis']['think']['global_avg']:.4f}\n"
            summary += f"NoThink Global Avg: {results['node_analysis']['nothink']['global_avg']:.4f}\n\n"
        
        # 边重要性统计
        summary += "Edge Importance:\n"
        think_edge_avg = results['edge_analysis']['think']['class_avg']
        nothink_edge_avg = results['edge_analysis']['nothink']['class_avg']
        
        if think_edge_avg and nothink_edge_avg:
            summary += f"Think Class Avg: {np.mean(think_edge_avg):.4f}±{np.std(think_edge_avg):.4f}\n"
            summary += f"NoThink Class Avg: {np.mean(nothink_edge_avg):.4f}±{np.std(nothink_edge_avg):.4f}\n"
            summary += f"Think Global Avg: {results['edge_analysis']['think']['global_avg']:.4f}\n"
            summary += f"NoThink Global Avg: {results['edge_analysis']['nothink']['global_avg']:.4f}\n"
        
        return summary
    
    def save_analysis_results(self, analysis_results, output_dir):
        """保存分析结果到文件"""
        results_file = os.path.join(output_dir, 'importance_analysis_results.json')
        
        # 转换numpy数组为列表以便JSON序列化，并确保所有数值都是Python原生类型
        json_results = {}
        for analysis_type in ['node_analysis', 'edge_analysis']:
            json_results[analysis_type] = {}
            for category in ['think', 'nothink']:
                # 转换class_avg列表中的numpy类型为Python float
                class_avg_list = analysis_results[analysis_type][category]['class_avg']
                if isinstance(class_avg_list, list):
                    class_avg_list = [float(x) for x in class_avg_list]
                else:
                    class_avg_list = []
                
                # 转换global_avg为Python float
                global_avg_val = analysis_results[analysis_type][category]['global_avg']
                if hasattr(global_avg_val, 'item'):  # numpy scalar
                    global_avg_val = float(global_avg_val.item())
                else:
                    global_avg_val = float(global_avg_val)
                
                json_results[analysis_type][category] = {
                    'class_avg': class_avg_list,
                    'global_avg': global_avg_val
                }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"分析结果已保存到: {results_file}")
    
    def create_network_visualizations(self, output_dir):
        """
        创建真正的网络可视化图，显示think、nothink和combined的重要性
        """
        print("创建网络可视化图...")
        
        if not self.avg_matrices:
            print("未找到平均矩阵，请先运行compute_average_matrices()")
            return
        
        # 创建3个子图：think, nothink, combined
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        fig.suptitle('Network Importance Visualization: Think vs NoThink vs Combined', fontsize=16, fontweight='bold')
        
        categories = ['think', 'nothink', 'combined']
        titles = ['Think Networks', 'NoThink Networks', 'Combined Networks']
        
        for idx, (category, title) in enumerate(zip(categories, titles)):
            ax = axes[idx]
            
            if (self.avg_matrices[category]['node_importance'] is not None and 
                self.avg_matrices[category]['edge_importance'] is not None):
                
                try:
                    # 获取数据
                    node_importance = self.avg_matrices[category]['node_importance']
                    edge_importance = self.avg_matrices[category]['edge_importance']
                    
                    # 限制节点数量以提高可视化性能
                    max_nodes = 100
                    if len(node_importance) > max_nodes:
                        # 选择重要性最高的节点
                        top_node_indices = np.argsort(node_importance)[-max_nodes:]
                        node_importance = node_importance[top_node_indices]
                        
                        # 过滤边，只保留连接重要节点的边
                        mask = np.isin(edge_index[0], top_node_indices) & np.isin(edge_index[1], top_node_indices)
                        edge_index = edge_index[:, mask]
                        edge_importance = edge_importance[mask]
                        
                        # 重新映射节点索引
                        node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(top_node_indices)}
                        edge_index = np.array([[node_mapping[edge_index[0, i]], node_mapping[edge_index[1, i]]] 
                                             for i in range(edge_index.shape[1])]).T
                    
                    # 创建NetworkX图
                    G = nx.Graph()
                    G.add_nodes_from(range(len(node_importance)))
                    
                    # 添加边
                    for i in range(edge_index.shape[1]):
                        src, dst = edge_index[0, i], edge_index[1, i]
                        if src < len(node_importance) and dst < len(node_importance):
                            G.add_edge(src, dst, weight=edge_importance[i])
                    
                    # 计算布局
                    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
                    
                    # 节点可视化 - 大小反映重要性
                    node_importance_norm = (node_importance - node_importance.min()) / (node_importance.max() - node_importance.min() + 1e-8)
                    node_sizes = 50 + 500 * node_importance_norm  # 节点大小范围：50-550
                    
                    # 节点颜色 - 重要性越高越红
                    node_colors = plt.cm.Reds(node_importance_norm)
                    
                    # 绘制节点
                    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, 
                                         alpha=0.8, ax=ax)
                    
                    # 边可视化 - 粗细和颜色反映重要性
                    if len(edge_importance) > 0:
                        edge_importance_norm = (edge_importance - edge_importance.min()) / (edge_importance.max() - edge_importance.min() + 1e-8)
                        
                        # 只显示重要性较高的边以避免过于密集
                        threshold = np.percentile(edge_importance_norm, 70)  # 只显示前30%的边
                        important_edge_mask = edge_importance_norm >= threshold
                        
                        if np.any(important_edge_mask):
                            important_edges = [(edge_index[0, i], edge_index[1, i]) 
                                             for i in range(edge_index.shape[1]) if important_edge_mask[i]]
                            important_weights = edge_importance_norm[important_edge_mask]
                            
                            # 边的粗细：0.5-5.0
                            edge_widths = 0.5 + 4.5 * important_weights
                            # 边的颜色：重要性越高越深
                            edge_colors = plt.cm.Blues(important_weights)
                            
                            # 绘制边
                            for edge, width, color in zip(important_edges, edge_widths, edge_colors):
                                nx.draw_networkx_edges(G, pos, edgelist=[edge], width=width, 
                                                     edge_color=[color], alpha=0.7, ax=ax)
                    
                    # 添加最重要的节点标签
                    top_5_nodes = np.argsort(node_importance)[-5:]
                    labels = {node: str(node) for node in top_5_nodes}
                    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, 
                                          font_weight='bold', ax=ax)
                    
                    ax.set_title(f'{title}\n(Nodes: {len(G.nodes())}, Edges: {len(G.edges())})')
                    ax.axis('off')
                    
                    # 添加图例
                    ax.text(0.02, 0.98, f'Node size ∝ Importance\nEdge width ∝ Importance\nTop 30% edges shown', 
                           transform=ax.transAxes, verticalalignment='top', fontsize=8,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
                    
                except Exception as e:
                    print(f"创建{category}网络图时出错: {e}")
                    ax.text(0.5, 0.5, f'Error creating {category} network\n{str(e)[:50]}...', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{title} - Error')
                    ax.axis('off')
            else:
                ax.text(0.5, 0.5, f'No data available for {category}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{title} - No Data')
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'network_importance_visualization.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 创建差异可视化
        self._create_difference_visualization(output_dir)
    
    def _create_difference_visualization(self, output_dir):
        """
        创建think和nothink之间的差异可视化
        """
        print("创建差异可视化图...")
        
        think_node = self.avg_matrices['think']['node_importance']
        nothink_node = self.avg_matrices['nothink']['node_importance']
        think_edge = self.avg_matrices['think']['edge_importance']
        nothink_edge = self.avg_matrices['nothink']['edge_importance']
        
        if think_node is not None and nothink_node is not None:
            # 确保长度相同
            min_len = min(len(think_node), len(nothink_node))
            node_diff = think_node[:min_len] - nothink_node[:min_len]
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Importance Difference Analysis: Think - NoThink', fontsize=16, fontweight='bold')
            
            # 节点重要性差异分布
            ax1 = axes[0, 0]
            ax1.hist(node_diff, bins=30, alpha=0.7, color='purple')
            ax1.set_xlabel('Node Importance Difference (Think - NoThink)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Node Importance Difference Distribution')
            ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            
            # 节点重要性差异排序
            ax2 = axes[0, 1]
            sorted_indices = np.argsort(np.abs(node_diff))[-20:]  # 前20个差异最大的节点
            ax2.barh(range(len(sorted_indices)), node_diff[sorted_indices], 
                    color=['red' if x > 0 else 'blue' for x in node_diff[sorted_indices]])
            ax2.set_xlabel('Node Importance Difference')
            ax2.set_ylabel('Node Index')
            ax2.set_title('Top 20 Nodes with Largest Importance Difference')
            ax2.set_yticks(range(len(sorted_indices)))
            ax2.set_yticklabels(sorted_indices)
            
            if think_edge is not None and nothink_edge is not None:
                # 边重要性差异
                min_edge_len = min(len(think_edge), len(nothink_edge))
                edge_diff = think_edge[:min_edge_len] - nothink_edge[:min_edge_len]
                
                # 边重要性差异分布
                ax3 = axes[1, 0]
                ax3.hist(edge_diff, bins=30, alpha=0.7, color='orange')
                ax3.set_xlabel('Edge Importance Difference (Think - NoThink)')
                ax3.set_ylabel('Frequency')
                ax3.set_title('Edge Importance Difference Distribution')
                ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
                
                # 边重要性差异排序
                ax4 = axes[1, 1]
                sorted_edge_indices = np.argsort(np.abs(edge_diff))[-20:]
                ax4.barh(range(len(sorted_edge_indices)), edge_diff[sorted_edge_indices],
                        color=['red' if x > 0 else 'blue' for x in edge_diff[sorted_edge_indices]])
                ax4.set_xlabel('Edge Importance Difference')
                ax4.set_ylabel('Edge Index')
                ax4.set_title('Top 20 Edges with Largest Importance Difference')
                ax4.set_yticks(range(len(sorted_edge_indices)))
                ax4.set_yticklabels(sorted_edge_indices)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'importance_difference_analysis.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def find_top_important_elements(self, output_dir, top_k=10):
        """
        找到前k个最重要的节点和边
        
        Args:
            output_dir: 输出目录
            top_k: 返回前k个重要元素，默认10
        """
        print(f"正在查找前{top_k}个最重要的节点和边...")
        
        if not self.avg_matrices:
            print("未找到平均矩阵，请先运行compute_average_matrices()")
            return
        
        results = {}
        
        # 分析每个类别的前k个重要元素
        for category in ['think', 'nothink', 'combined']:
            results[category] = {}
            
            # 分析节点重要性
            if self.avg_matrices[category]['node_importance'] is not None:
                node_importance = self.avg_matrices[category]['node_importance']
                
                # 找到前k个重要节点
                top_node_indices = np.argsort(node_importance)[-top_k:][::-1]  # 降序排列
                top_node_values = node_importance[top_node_indices]
                
                results[category]['top_nodes'] = {
                    'indices': top_node_indices.tolist(),
                    'values': top_node_values.tolist(),
                    'mean_importance': float(np.mean(top_node_values)),
                    'std_importance': float(np.std(top_node_values))
                }
                
                print(f"{category} - 前{top_k}个重要节点:")
                for i, (idx, val) in enumerate(zip(top_node_indices, top_node_values)):
                    print(f"  第{i+1}名: 节点{idx}, 重要性={val:.6f}")
            
            # 分析边重要性
            if self.avg_matrices[category]['edge_importance'] is not None:
                edge_importance = self.avg_matrices[category]['edge_importance']
                
                # 找到前k个重要边
                top_edge_indices = np.argsort(edge_importance)[-top_k:][::-1]  # 降序排列
                top_edge_values = edge_importance[top_edge_indices]
                
                # 使用全连接网络的方法计算边连接信息
                edge_connections = []
                for edge_idx in top_edge_indices:
                    # 使用除整取余方法计算端点
                    src, dst = self.get_edge_endpoints_from_index(edge_idx)
                    
                    # 验证节点索引是否有效
                    if src < self.num_nodes and dst < self.num_nodes and src >= 0 and dst >= 0:
                        edge_connections.append([int(src), int(dst)])
                    else:
                        # 如果计算出的节点索引超出范围，记录错误
                        print(f"警告: 边{edge_idx}计算出的端点({src}, {dst})超出节点范围[0, {self.num_nodes-1}]")
                        edge_connections.append([None, None])
                
                results[category]['top_edges'] = {
                    'indices': top_edge_indices.tolist(),
                    'values': top_edge_values.tolist(),
                    'connections': edge_connections,
                    'mean_importance': float(np.mean(top_edge_values)),
                    'std_importance': float(np.std(top_edge_values))
                }
                
                print(f"{category} - 前{top_k}个重要边:")
                for i, (idx, val, conn) in enumerate(zip(top_edge_indices, top_edge_values, edge_connections)):
                    if conn[0] is not None and conn[1] is not None:
                        print(f"  第{i+1}名: 边{idx} ({conn[0]}->{conn[1]}), 重要性={val:.6f}")
                    else:
                        print(f"  第{i+1}名: 边{idx}, 重要性={val:.6f}")
            
            print()
        
        # 保存结果到文件
        self._save_top_elements_results(results, output_dir, top_k)
        
        # 创建可视化
        self._visualize_top_elements(results, output_dir, top_k)
        
        return results
    
    def _save_top_elements_results(self, results, output_dir, top_k):
        """保存前k个重要元素结果到文件"""
        results_file = os.path.join(output_dir, f'top_{top_k}_important_elements.json')
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 创建可读的文本报告
        report_file = os.path.join(output_dir, f'top_{top_k}_important_elements_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"前{top_k}个最重要元素分析报告\n")
            f.write("=" * 50 + "\n\n")
            
            for category in ['think', 'nothink', 'combined']:
                f.write(f"{category.upper()} 类别分析:\n")
                f.write("-" * 30 + "\n")
                
                if 'top_nodes' in results[category]:
                    f.write(f"前{top_k}个重要节点:\n")
                    nodes = results[category]['top_nodes']
                    for i, (idx, val) in enumerate(zip(nodes['indices'], nodes['values'])):
                        f.write(f"  第{i+1}名: 节点{idx}, 重要性={val:.6f}\n")
                    f.write(f"  平均重要性: {nodes['mean_importance']:.6f}\n")
                    f.write(f"  标准差: {nodes['std_importance']:.6f}\n\n")
                
                if 'top_edges' in results[category]:
                    f.write(f"前{top_k}个重要边:\n")
                    edges = results[category]['top_edges']
                    for i, (idx, val, conn) in enumerate(zip(edges['indices'], edges['values'], edges['connections'])):
                        if conn[0] is not None and conn[1] is not None:
                            f.write(f"  第{i+1}名: 边{idx} ({conn[0]}->{conn[1]}), 重要性={val:.6f}\n")
                        else:
                            f.write(f"  第{i+1}名: 边{idx}, 重要性={val:.6f}\n")
                    f.write(f"  平均重要性: {edges['mean_importance']:.6f}\n")
                    f.write(f"  标准差: {edges['std_importance']:.6f}\n\n")
                
                f.write("\n")
        
        print(f"前{top_k}个重要元素结果已保存到:")
        print(f"  JSON格式: {results_file}")
        print(f"  文本报告: {report_file}")
    
    def _visualize_top_elements(self, results, output_dir, top_k):
        """可视化前k个重要元素"""
        print(f"创建前{top_k}个重要元素可视化...")
        
        # 创建对比图表
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Top {top_k} Most Important Elements Comparison', fontsize=16, fontweight='bold')
        
        categories = ['think', 'nothink', 'combined']
        colors = {'think': '#2E8B57', 'nothink': '#DC143C', 'combined': '#4169E1'}
        
        # 节点重要性对比
        for i, category in enumerate(categories):
            ax = axes[0, i]
            if 'top_nodes' in results[category]:
                nodes = results[category]['top_nodes']
                indices = nodes['indices']
                values = nodes['values']
                
                bars = ax.bar(range(len(indices)), values, color=colors[category], alpha=0.7)
                ax.set_xlabel('Rank')
                ax.set_ylabel('Node Importance')
                ax.set_title(f'{category.capitalize()} - Top {top_k} Nodes')
                ax.set_xticks(range(len(indices)))
                ax.set_xticklabels([f'{i+1}\n(Node {idx})' for i, idx in enumerate(indices)], rotation=45)
                
                # 在柱子上添加数值
                for bar, value in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                           f'{value:.4f}', ha='center', va='bottom', fontsize=8)
        
        # 边重要性对比
        for i, category in enumerate(categories):
            ax = axes[1, i]
            if 'top_edges' in results[category]:
                edges = results[category]['top_edges']
                indices = edges['indices']
                values = edges['values']
                connections = edges['connections']
                
                bars = ax.bar(range(len(indices)), values, color=colors[category], alpha=0.7)
                ax.set_xlabel('Rank')
                ax.set_ylabel('Edge Importance')
                ax.set_title(f'{category.capitalize()} - Top {top_k} Edges')
                
                # 创建标签
                labels = []
                for i, (idx, conn) in enumerate(zip(indices, connections)):
                    if conn[0] is not None and conn[1] is not None:
                        labels.append(f'{i+1}\n({conn[0]}-{conn[1]})')
                    else:
                        labels.append(f'{i+1}\n(Edge {idx})')
                
                ax.set_xticks(range(len(indices)))
                ax.set_xticklabels(labels, rotation=45)
                
                # 在柱子上添加数值
                for bar, value in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                           f'{value:.4f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'top_{top_k}_elements_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 创建重要性分布对比图
        self._create_importance_distribution_comparison(results, output_dir, top_k)
    
    def _create_importance_distribution_comparison(self, results, output_dir, top_k):
        """创建重要性分布对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Top {top_k} Elements Importance Distribution Comparison', fontsize=16, fontweight='bold')
        
        colors = {'think': '#2E8B57', 'nothink': '#DC143C', 'combined': '#4169E1'}
        
        # 节点重要性分布对比
        ax1 = axes[0, 0]
        node_data = []
        node_labels = []
        for category in ['think', 'nothink', 'combined']:
            if 'top_nodes' in results[category]:
                node_data.append(results[category]['top_nodes']['values'])
                node_labels.append(category.capitalize())
        
        if node_data:
            bp = ax1.boxplot(node_data, labels=node_labels, patch_artist=True)
            for patch, category in zip(bp['boxes'], ['think', 'nothink', 'combined'][:len(bp['boxes'])]):
                patch.set_facecolor(colors[category])
                patch.set_alpha(0.7)
        
        ax1.set_ylabel('Node Importance')
        ax1.set_title(f'Top {top_k} Node Importance Distribution')
        
        # 边重要性分布对比
        ax2 = axes[0, 1]
        edge_data = []
        edge_labels = []
        for category in ['think', 'nothink', 'combined']:
            if 'top_edges' in results[category]:
                edge_data.append(results[category]['top_edges']['values'])
                edge_labels.append(category.capitalize())
        
        if edge_data:
            bp = ax2.boxplot(edge_data, labels=edge_labels, patch_artist=True)
            for patch, category in zip(bp['boxes'], ['think', 'nothink', 'combined'][:len(bp['boxes'])]):
                patch.set_facecolor(colors[category])
                patch.set_alpha(0.7)
        
        ax2.set_ylabel('Edge Importance')
        ax2.set_title(f'Top {top_k} Edge Importance Distribution')
        
        # 平均重要性对比
        ax3 = axes[1, 0]
        categories = []
        node_means = []
        edge_means = []
        
        for category in ['think', 'nothink', 'combined']:
            if 'top_nodes' in results[category] or 'top_edges' in results[category]:
                categories.append(category.capitalize())
                node_means.append(results[category].get('top_nodes', {}).get('mean_importance', 0))
                edge_means.append(results[category].get('top_edges', {}).get('mean_importance', 0))
        
        if categories:
            x = np.arange(len(categories))
            width = 0.35
            
            ax3.bar(x - width/2, node_means, width, label='Nodes', alpha=0.7)
            ax3.bar(x + width/2, edge_means, width, label='Edges', alpha=0.7)
            
            ax3.set_xlabel('Category')
            ax3.set_ylabel('Mean Importance')
            ax3.set_title(f'Mean Importance of Top {top_k} Elements')
            ax3.set_xticks(x)
            ax3.set_xticklabels(categories)
            ax3.legend()
        
        # 统计摘要
        ax4 = axes[1, 1]
        summary_text = f"Top {top_k} Elements Summary\n" + "="*25 + "\n\n"
        
        for category in ['think', 'nothink', 'combined']:
            summary_text += f"{category.upper()}:\n"
            if 'top_nodes' in results[category]:
                nodes = results[category]['top_nodes']
                summary_text += f"  Nodes: {nodes['mean_importance']:.4f}±{nodes['std_importance']:.4f}\n"
            if 'top_edges' in results[category]:
                edges = results[category]['top_edges']
                summary_text += f"  Edges: {edges['mean_importance']:.4f}±{edges['std_importance']:.4f}\n"
            summary_text += "\n"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax4.set_title('Statistical Summary')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'top_{top_k}_distribution_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_complete_analysis(self, output_dir, layer_idx=None, find_top_k=10):
        """运行完整的重要性分析"""
        print("开始运行完整的重要性分析...")
        
        # 1. 加载数据
        self.load_explanation_data(layer_idx)
        
        # 2. 计算类内平均和全局平均
        analysis_results = self.compute_class_and_global_averages()
        
        # 3. 计算平均矩阵
        self.compute_average_matrices()
        
        # 4. 保存平均矩阵
        self.save_average_matrices(output_dir)
        
        # 5. 创建可视化
        self.create_comparison_visualizations(analysis_results, output_dir)
        
        # 6. 创建网络可视化
        self.create_network_visualizations(output_dir)
        
        # 7. 找到前k个重要元素
        top_elements_results = self.find_top_important_elements(output_dir, find_top_k)
        
        # 8. 保存结果
        self.save_analysis_results(analysis_results, output_dir)
        
        print("重要性分析完成！")
        print("生成的文件包括:")
        print("- 统计分析图表: node_importance_analysis.png, edge_importance_analysis.png, comprehensive_comparison.png")
        print("- 网络可视化图: network_importance_visualization.png, importance_difference_analysis.png")
        print("- 平均矩阵文件: think_avg_node_importance.npy, nothink_avg_node_importance.npy 等")
        print("- 分析结果: importance_analysis_results.json")
        print(f"- 前{find_top_k}个重要元素: top_{find_top_k}_important_elements.json, top_{find_top_k}_important_elements_report.txt")
        print(f"- 前{find_top_k}个重要元素可视化: top_{find_top_k}_elements_comparison.png, top_{find_top_k}_distribution_comparison.png")
        
        return analysis_results, top_elements_results

    @staticmethod
    def find_top_elements_from_saved_matrices(matrices_dir, output_dir=None, top_k=10, num_nodes=1024):
        """
        从已保存的平均重要性矩阵中直接找到前k个最重要的节点和边
        
        Args:
            matrices_dir: 保存矩阵的目录路径
            output_dir: 输出目录，如果为None则使用matrices_dir
            top_k: 返回前k个重要元素，默认10
            num_nodes: 全连接网络的节点数，默认1024
        """
        if output_dir is None:
            output_dir = matrices_dir
            
        print(f"从已保存的矩阵中查找前{top_k}个最重要的节点和边...")
        
        results = {}
        
        def get_edge_endpoints_from_index(edge_idx):
            """计算边端点（全连接网络）"""
            src = edge_idx // num_nodes
            dst = edge_idx % num_nodes
            return src, dst
        
        # 检查并加载已保存的矩阵
        for category in ['think', 'nothink', 'combined']:
            results[category] = {}
            
            # 加载节点重要性矩阵
            node_file = os.path.join(matrices_dir, f'{category}_avg_node_importance.npy')
            if os.path.exists(node_file):
                try:
                    node_importance = np.load(node_file)
                    print(f"加载 {category} 节点重要性矩阵: {node_importance.shape}")
                    
                    # 找到前k个重要节点
                    top_node_indices = np.argsort(node_importance)[-top_k:][::-1]  # 降序排列
                    top_node_values = node_importance[top_node_indices]
                    
                    results[category]['top_nodes'] = {
                        'indices': top_node_indices.tolist(),
                        'values': top_node_values.tolist(),
                        'mean_importance': float(np.mean(top_node_values)),
                        'std_importance': float(np.std(top_node_values))
                    }
                    
                    print(f"{category} - 前{top_k}个重要节点:")
                    for i, (idx, val) in enumerate(zip(top_node_indices, top_node_values)):
                        print(f"  第{i+1}名: 节点{idx}, 重要性={val:.6f}")
                        
                except Exception as e:
                    print(f"加载{category}节点重要性矩阵失败: {e}")
            
            # 加载边重要性矩阵
            edge_file = os.path.join(matrices_dir, f'{category}_avg_edge_importance.npy')
            
            if os.path.exists(edge_file):
                try:
                    edge_importance = np.load(edge_file)
                    print(f"加载 {category} 边重要性矩阵: {edge_importance.shape}")
                    
                    # 找到前k个重要边
                    top_edge_indices = np.argsort(edge_importance)[-top_k:][::-1]  # 降序排列
                    top_edge_values = edge_importance[top_edge_indices]
                    
                    # 使用全连接网络的方法计算边连接信息
                    edge_connections = []
                    for edge_idx in top_edge_indices:
                        # 使用除整取余方法计算端点
                        src, dst = get_edge_endpoints_from_index(edge_idx)
                        
                        # 验证节点索引是否有效
                        if src < num_nodes and dst < num_nodes and src >= 0 and dst >= 0:
                            edge_connections.append([int(src), int(dst)])
                        else:
                            # 如果计算出的节点索引超出范围，记录错误
                            print(f"警告: 边{edge_idx}计算出的端点({src}, {dst})超出节点范围[0, {num_nodes-1}]")
                            edge_connections.append([None, None])
                    
                    results[category]['top_edges'] = {
                        'indices': top_edge_indices.tolist(),
                        'values': top_edge_values.tolist(),
                        'connections': edge_connections,
                        'mean_importance': float(np.mean(top_edge_values)),
                        'std_importance': float(np.std(top_edge_values))
                    }
                    
                    print(f"{category} - 前{top_k}个重要边:")
                    for i, (idx, val, conn) in enumerate(zip(top_edge_indices, top_edge_values, edge_connections)):
                        if conn[0] is not None and conn[1] is not None:
                            print(f"  第{i+1}名: 边{idx} ({conn[0]}->{conn[1]}), 重要性={val:.6f}")
                        else:
                            print(f"  第{i+1}名: 边{idx}, 重要性={val:.6f}")
                            
                except Exception as e:
                    print(f"加载{category}边重要性矩阵失败: {e}")
            
            print()
        
        # 保存结果
        ImportanceAnalyzer._save_top_elements_results_static(results, output_dir, top_k)
        
        # 创建可视化
        ImportanceAnalyzer._visualize_top_elements_static(results, output_dir, top_k)
        
        return results
    
    @staticmethod
    def _save_top_elements_results_static(results, output_dir, top_k):
        """静态方法：保存前k个重要元素结果到文件"""
        os.makedirs(output_dir, exist_ok=True)
        
        results_file = os.path.join(output_dir, f'top_{top_k}_important_elements.json')
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 创建可读的文本报告
        report_file = os.path.join(output_dir, f'top_{top_k}_important_elements_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"前{top_k}个最重要元素分析报告\n")
            f.write("=" * 50 + "\n\n")
            
            for category in ['think', 'nothink', 'combined']:
                f.write(f"{category.upper()} 类别分析:\n")
                f.write("-" * 30 + "\n")
                
                if 'top_nodes' in results[category]:
                    f.write(f"前{top_k}个重要节点:\n")
                    nodes = results[category]['top_nodes']
                    for i, (idx, val) in enumerate(zip(nodes['indices'], nodes['values'])):
                        f.write(f"  第{i+1}名: 节点{idx}, 重要性={val:.6f}\n")
                    f.write(f"  平均重要性: {nodes['mean_importance']:.6f}\n")
                    f.write(f"  标准差: {nodes['std_importance']:.6f}\n\n")
                
                if 'top_edges' in results[category]:
                    f.write(f"前{top_k}个重要边:\n")
                    edges = results[category]['top_edges']
                    for i, (idx, val, conn) in enumerate(zip(edges['indices'], edges['values'], edges['connections'])):
                        if conn[0] is not None and conn[1] is not None:
                            f.write(f"  第{i+1}名: 边{idx} ({conn[0]}->{conn[1]}), 重要性={val:.6f}\n")
                        else:
                            f.write(f"  第{i+1}名: 边{idx}, 重要性={val:.6f}\n")
                    f.write(f"  平均重要性: {edges['mean_importance']:.6f}\n")
                    f.write(f"  标准差: {edges['std_importance']:.6f}\n\n")
                
                f.write("\n")
        
        print(f"前{top_k}个重要元素结果已保存到:")
        print(f"  JSON格式: {results_file}")
        print(f"  文本报告: {report_file}")
    
    @staticmethod
    def _visualize_top_elements_static(results, output_dir, top_k):
        """静态方法：可视化前k个重要元素"""
        print(f"创建前{top_k}个重要元素可视化...")
        
        # 创建对比图表
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Top {top_k} Most Important Elements Comparison', fontsize=16, fontweight='bold')
        
        categories = ['think', 'nothink', 'combined']
        colors = {'think': '#2E8B57', 'nothink': '#DC143C', 'combined': '#4169E1'}
        
        # 节点重要性对比
        for i, category in enumerate(categories):
            ax = axes[0, i]
            if 'top_nodes' in results[category]:
                nodes = results[category]['top_nodes']
                indices = nodes['indices']
                values = nodes['values']
                
                bars = ax.bar(range(len(indices)), values, color=colors[category], alpha=0.7)
                ax.set_xlabel('Rank')
                ax.set_ylabel('Node Importance')
                ax.set_title(f'{category.capitalize()} - Top {top_k} Nodes')
                ax.set_xticks(range(len(indices)))
                ax.set_xticklabels([f'{i+1}\n(Node {idx})' for i, idx in enumerate(indices)], rotation=45)
                
                # 在柱子上添加数值
                for bar, value in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                           f'{value:.4f}', ha='center', va='bottom', fontsize=8)
        
        # 边重要性对比
        for i, category in enumerate(categories):
            ax = axes[1, i]
            if 'top_edges' in results[category]:
                edges = results[category]['top_edges']
                indices = edges['indices']
                values = edges['values']
                connections = edges['connections']
                
                bars = ax.bar(range(len(indices)), values, color=colors[category], alpha=0.7)
                ax.set_xlabel('Rank')
                ax.set_ylabel('Edge Importance')
                ax.set_title(f'{category.capitalize()} - Top {top_k} Edges')
                
                # 创建标签
                labels = []
                for i, (idx, conn) in enumerate(zip(indices, connections)):
                    if conn[0] is not None and conn[1] is not None:
                        labels.append(f'{i+1}\n({conn[0]}-{conn[1]})')
                    else:
                        labels.append(f'{i+1}\n(Edge {idx})')
                
                ax.set_xticks(range(len(indices)))
                ax.set_xticklabels(labels, rotation=45)
                
                # 在柱子上添加数值
                for bar, value in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                           f'{value:.4f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'top_{top_k}_elements_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 创建重要性分布对比图
        ImportanceAnalyzer._create_importance_distribution_comparison_static(results, output_dir, top_k)
    
    @staticmethod
    def _create_importance_distribution_comparison_static(results, output_dir, top_k):
        """静态方法：创建重要性分布对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Top {top_k} Elements Importance Distribution Comparison', fontsize=16, fontweight='bold')
        
        colors = {'think': '#2E8B57', 'nothink': '#DC143C', 'combined': '#4169E1'}
        
        # 节点重要性分布对比
        ax1 = axes[0, 0]
        node_data = []
        node_labels = []
        for category in ['think', 'nothink', 'combined']:
            if 'top_nodes' in results[category]:
                node_data.append(results[category]['top_nodes']['values'])
                node_labels.append(category.capitalize())
        
        if node_data:
            bp = ax1.boxplot(node_data, labels=node_labels, patch_artist=True)
            for patch, category in zip(bp['boxes'], ['think', 'nothink', 'combined'][:len(bp['boxes'])]):
                patch.set_facecolor(colors[category])
                patch.set_alpha(0.7)
        
        ax1.set_ylabel('Node Importance')
        ax1.set_title(f'Top {top_k} Node Importance Distribution')
        
        # 边重要性分布对比
        ax2 = axes[0, 1]
        edge_data = []
        edge_labels = []
        for category in ['think', 'nothink', 'combined']:
            if 'top_edges' in results[category]:
                edge_data.append(results[category]['top_edges']['values'])
                edge_labels.append(category.capitalize())
        
        if edge_data:
            bp = ax2.boxplot(edge_data, labels=edge_labels, patch_artist=True)
            for patch, category in zip(bp['boxes'], ['think', 'nothink', 'combined'][:len(bp['boxes'])]):
                patch.set_facecolor(colors[category])
                patch.set_alpha(0.7)
        
        ax2.set_ylabel('Edge Importance')
        ax2.set_title(f'Top {top_k} Edge Importance Distribution')
        
        # 平均重要性对比
        ax3 = axes[1, 0]
        categories = []
        node_means = []
        edge_means = []
        
        for category in ['think', 'nothink', 'combined']:
            if 'top_nodes' in results[category] or 'top_edges' in results[category]:
                categories.append(category.capitalize())
                node_means.append(results[category].get('top_nodes', {}).get('mean_importance', 0))
                edge_means.append(results[category].get('top_edges', {}).get('mean_importance', 0))
        
        if categories:
            x = np.arange(len(categories))
            width = 0.35
            
            ax3.bar(x - width/2, node_means, width, label='Nodes', alpha=0.7)
            ax3.bar(x + width/2, edge_means, width, label='Edges', alpha=0.7)
            
            ax3.set_xlabel('Category')
            ax3.set_ylabel('Mean Importance')
            ax3.set_title(f'Mean Importance of Top {top_k} Elements')
            ax3.set_xticks(x)
            ax3.set_xticklabels(categories)
            ax3.legend()
        
        # 统计摘要
        ax4 = axes[1, 1]
        summary_text = f"Top {top_k} Elements Summary\n" + "="*25 + "\n\n"
        
        for category in ['think', 'nothink', 'combined']:
            summary_text += f"{category.upper()}:\n"
            if 'top_nodes' in results[category]:
                nodes = results[category]['top_nodes']
                summary_text += f"  Nodes: {nodes['mean_importance']:.4f}±{nodes['std_importance']:.4f}\n"
            if 'top_edges' in results[category]:
                edges = results[category]['top_edges']
                summary_text += f"  Edges: {edges['mean_importance']:.4f}±{edges['std_importance']:.4f}\n"
            summary_text += "\n"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax4.set_title('Statistical Summary')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'top_{top_k}_distribution_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='重要性分析可视化')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='explanation结果目录路径')
    parser.add_argument('--output_dir', type=str, default='./importance_analysis_output',
                       help='输出目录路径')
    parser.add_argument('--layer', type=int, default=14,
                       help='指定分析的layer（默认分析所有layer）')
    parser.add_argument('--top_k', type=int, default=10,
                       help='查找前k个最重要的元素（默认10）')
    parser.add_argument('--num_nodes', type=int, default=1024,
                       help='全连接网络的节点数（默认1024）')
    
    args = parser.parse_args()
    
    # 创建分析器并运行分析
    analyzer = ImportanceAnalyzer(args.results_dir, args.num_nodes)
    results, top_elements = analyzer.run_complete_analysis(args.output_dir, args.layer, args.top_k)
    
    print(f"分析结果已保存到: {args.output_dir}")


def find_top_elements_only():
    """独立函数：仅查找前k个重要元素，不重新计算矩阵"""
    parser = argparse.ArgumentParser(description='从已保存的矩阵中查找前k个重要元素')
    parser.add_argument('--matrices_dir', type=str, required=True,
                       help='已保存矩阵的目录路径')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录路径（默认使用matrices_dir）')
    parser.add_argument('--top_k', type=int, default=10,
                       help='查找前k个最重要的元素（默认10）')
    parser.add_argument('--num_nodes', type=int, default=1024,
                       help='全连接网络的节点数（默认1024）')
    
    args = parser.parse_args()
    
    # 直接从已保存的矩阵中查找前k个重要元素
    results = ImportanceAnalyzer.find_top_elements_from_saved_matrices(
        args.matrices_dir, args.output_dir, args.top_k, args.num_nodes
    )
    
    output_dir = args.output_dir if args.output_dir else args.matrices_dir
    print(f"前{args.top_k}个重要元素分析结果已保存到: {output_dir}")


if __name__ == "__main__":
    import sys
    
    # 检查是否只是要查找前k个重要元素
    if len(sys.argv) > 1 and sys.argv[1] == 'find_top_only':
        # 移除'find_top_only'参数，让argparse正常解析剩余参数
        sys.argv.pop(1)
        find_top_elements_only()
    else:
        main() 