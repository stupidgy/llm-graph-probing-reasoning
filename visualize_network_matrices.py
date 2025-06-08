#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import networkx as nx
import matplotlib

# 设置matplotlib参数
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Bitstream Vera Sans', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')

def load_network_matrix(file_path):
    """加载网络矩阵文件"""
    try:
        matrix = np.load(file_path)
        print(f"Successfully loaded file: {file_path}")
        print(f"Matrix shape: {matrix.shape}")
        return matrix
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def print_matrix_stats(matrix, name="Matrix"):
    """打印矩阵的详细统计信息"""
    print(f"\n{name} statistics:")
    print(f"  Shape: {matrix.shape}")
    print(f"  Max value: {np.max(matrix):.6f}")
    print(f"  Min value: {np.min(matrix):.6f}")
    print(f"  Mean value: {np.mean(matrix):.6f}")
    print(f"  Standard deviation: {np.std(matrix):.6f}")
    print(f"  Median: {np.median(matrix):.6f}")
    print(f"  Mean absolute value: {np.mean(np.abs(matrix)):.6f}")
    
    # 计算不同阈值下的非零值占比
    for threshold in [0, 0.001, 0.01, 0.05, 0.1]:
        non_zero = np.sum(np.abs(matrix) > threshold)
        print(f"  Values with |x| > {threshold}: {non_zero/matrix.size*100:.2f}%")
    
    # 计算分位数
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    values = np.percentile(matrix, percentiles)
    for p, v in zip(percentiles, values):
        print(f"  {p}th percentile: {v:.6f}")

def plot_important_edges(matrix, title, output_path, top_n=50, min_weight=0.01, percent=None, show_node_ids=True, max_node_labels=50, font_size=12):
    """绘制网络中最重要的边
    
    参数:
        matrix: 网络矩阵
        title: 图表标题
        output_path: 输出路径前缀
        top_n: 要显示的边数量（当percent为None时使用）
        min_weight: 最小权重阈值（绝对值）
        percent: 要显示的边的百分比（如果提供，则覆盖top_n）
        show_node_ids: 是否显示节点ID
        max_node_labels: 最多显示多少个节点标签（避免拥挤）
        font_size: 节点标签的字体大小
    """
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 将矩阵转换为边列表格式
    n = matrix.shape[0]
    all_edges = []
    
    # 收集所有边（只考虑下三角，避免重复）
    for i in range(n):
        for j in range(i+1, n):
            weight = matrix[i, j]
            if abs(weight) > 0:  # 只收集非零边
                all_edges.append((i, j, abs(weight), weight))  # 存储绝对值(用于排序)和原始值(用于颜色)
    
    # 检查是否有边
    if not all_edges:
        print(f"Warning: No non-zero edges found")
        return
    
    # 按权重绝对值排序边
    all_edges.sort(key=lambda x: x[2], reverse=True)
    
    # 根据百分比或绝对数量选择边
    if percent is not None:
        # 使用百分比选择边
        if percent <= 0 or percent > 100:
            print(f"Warning: Percentage must be in range (0, 100], current value: {percent}")
            percent = 20  # 默认使用20%
        
        num_edges = int(len(all_edges) * percent / 100)
        num_edges = max(1, num_edges)  # 至少选择一条边
        important_edges = all_edges[:num_edges]
        selection_method = f"Top {percent}% ({num_edges}/{len(all_edges)})"
    else:
        # 先基于阈值过滤
        threshold_edges = [e for e in all_edges if e[2] >= min_weight]
        
        # 然后选择前top_n个
        important_edges = threshold_edges[:top_n] if top_n < len(threshold_edges) else threshold_edges
        selection_method = f"Top {len(important_edges)} (Weight >= {min_weight})"
    
    print(f"Selected {len(important_edges)} important edges, method: {selection_method}")
    
    if len(important_edges) == 0:
        print(f"Warning: No edges match the criteria")
        return
    
    # 创建图形
    plt.figure(figsize=(18, 16))  # 增大图形尺寸以适应较大的标签
    G = nx.Graph()
    
    # 添加节点
    nodes = set()
    for edge in important_edges:
        nodes.add(edge[0])
        nodes.add(edge[1])
    
    nodes_list = list(nodes)
    
    # 确定节点位置 - 使用圆形布局
    pos = nx.circular_layout(nodes_list)
    
    # 添加边，正相关为红色，负相关为蓝色
    pos_edges = [(u, v) for u, v, _, w in important_edges if w > 0]
    neg_edges = [(u, v) for u, v, _, w in important_edges if w < 0]
    
    # 提取边权重用于线宽
    pos_weights = [abs(w) * 5 for _, _, _, w in important_edges if w > 0]  # 乘以5增加可见度
    neg_weights = [abs(w) * 5 for _, _, _, w in important_edges if w < 0]
    
    # 绘制节点 - 增大节点尺寸以适应较大的标签
    nx.draw_networkx_nodes(G, pos, nodelist=nodes_list, node_size=200, node_color='lightgray')
    
    # 绘制正相关边
    if pos_edges:
        nx.draw_networkx_edges(G, pos, edgelist=pos_edges, width=pos_weights, 
                               edge_color='red', alpha=0.7)
    
    # 绘制负相关边
    if neg_edges:
        nx.draw_networkx_edges(G, pos, edgelist=neg_edges, width=neg_weights, 
                               edge_color='blue', alpha=0.7)
    
    # 添加节点标签
    if show_node_ids:
        # 如果节点太多，只标记部分节点以避免过度拥挤
        if len(nodes) > max_node_labels:
            # 计算节点重要性 - 将与重要边相连的节点视为重要节点
            node_importance = {}
            for node in nodes:
                # 计算与该节点相连的边的权重总和
                connected_edges = [e for e in important_edges if node in (e[0], e[1])]
                importance = sum(e[2] for e in connected_edges)
                node_importance[node] = importance
            
            # 选择最重要的节点进行标记
            important_nodes = sorted(node_importance.keys(), 
                                    key=lambda x: node_importance[x], 
                                    reverse=True)[:max_node_labels]
            
            # 创建标签字典，只包含重要节点
            labels = {node: str(node) for node in important_nodes}
            print(f"Showing labels for top {len(important_nodes)} important nodes out of {len(nodes)}")
        else:
            # 所有节点都标记
            labels = {node: str(node) for node in nodes}
        
        # 绘制节点标签 - 使用更大的字体和更大的背景框
        nx.draw_networkx_labels(G, pos, labels, font_size=font_size, font_color='black',
                               font_weight='bold', font_family='sans-serif',
                               bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', linewidth=0.5, pad=3))
    
    # 添加标题
    plt.title(f"{title} - {selection_method} Connections", fontsize=16)
    plt.axis('off')  # 隐藏坐标轴
    plt.tight_layout()
    
    # 保存图像 - 确保使用正确的文件扩展名
    output_path_net = f"{output_path}_network.png"
    plt.savefig(output_path_net, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制分布直方图
    plt.figure(figsize=(10, 6))
    sns.histplot(matrix.flatten(), bins=100, kde=True, color='steelblue')
    plt.title(f"Value Distribution - {title}", fontsize=14)
    plt.xlabel("Correlation Value")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_path}_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved network visualization to: {output_path_net}")
    print(f"Saved distribution histogram to: {output_path}_distribution.png")

def main():
    parser = argparse.ArgumentParser(description='Visualize Network Matrices as Network Graphs')
    parser.add_argument('--think_network', type=str, default='network_analysis_results/think_average_network.npy',
                        help='Path to Think model average network NPY file')
    parser.add_argument('--nothink_network', type=str, default='network_analysis_results/nothink_average_network.npy',
                        help='Path to NoThink model average network NPY file')
    parser.add_argument('--difference_network', type=str, default='network_analysis_results/difference_network.npy',
                        help='Path to difference network NPY file')
    parser.add_argument('--output_dir', type=str, default='network_visualizations',
                        help='Output directory')
    parser.add_argument('--top_edges', type=int, default=50,
                        help='Number of important edges to highlight')
    parser.add_argument('--min_weight', type=float, default=0,
                        help='Minimum absolute weight threshold for important edges')
    parser.add_argument('--percent', type=float, default=20.0,
                        help='Percentage of important edges to highlight, e.g., 20 means top 20%')
    parser.add_argument('--use_percent', action='store_true',
                        help='Use percentage to select important edges instead of fixed count and threshold')
    parser.add_argument('--show_node_ids', action='store_true', default=True,
                        help='Show node IDs in network visualization')
    parser.add_argument('--hide_node_ids', action='store_true',
                        help='Hide node IDs in network visualization')
    parser.add_argument('--max_node_labels', type=int, default=50,
                        help='Maximum number of node labels to show (to avoid overcrowding)')
    parser.add_argument('--font_size', type=int, default=20,
                        help='Font size for node labels (default: 12)')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 读取网络矩阵
    think_network = load_network_matrix(args.think_network)
    nothink_network = load_network_matrix(args.nothink_network)
    difference_network = load_network_matrix(args.difference_network)
    
    # 打印统计信息
    if think_network is not None:
        print_matrix_stats(think_network, "Think average network")
        
    if nothink_network is not None:
        print_matrix_stats(nothink_network, "NoThink average network")
        
    if difference_network is not None:
        print_matrix_stats(difference_network, "Difference network")
    
    # 准备选择重要边的参数
    percent_arg = args.percent if args.use_percent else None
    
    # 确定是否显示节点ID
    show_node_ids = args.show_node_ids and not args.hide_node_ids
    
    # 绘制网络图
    if think_network is not None:
        # 绘制重要边
        plot_important_edges(
            think_network,
            "Think Model Average Correlation Network",
            os.path.join(args.output_dir, "think"),
            top_n=args.top_edges,
            min_weight=args.min_weight,
            percent=percent_arg,
            show_node_ids=show_node_ids,
            max_node_labels=args.max_node_labels,
            font_size=args.font_size
        )
    
    if nothink_network is not None:
        # 绘制重要边
        plot_important_edges(
            nothink_network,
            "NoThink Model Average Correlation Network",
            os.path.join(args.output_dir, "nothink"),
            top_n=args.top_edges,
            min_weight=args.min_weight,
            percent=percent_arg,
            show_node_ids=show_node_ids,
            max_node_labels=args.max_node_labels,
            font_size=args.font_size
        )
    
    if difference_network is not None:
        # 绘制重要边
        plot_important_edges(
            difference_network,
            "Difference Network (Think - NoThink)",
            os.path.join(args.output_dir, "difference"),
            top_n=args.top_edges,
            min_weight=args.min_weight,
            percent=percent_arg,
            show_node_ids=show_node_ids,
            max_node_labels=args.max_node_labels,
            font_size=args.font_size
        )
    
    print(f"\nVisualization completed. Network graphs and distribution histograms saved in {args.output_dir} directory")

if __name__ == "__main__":
    main() 