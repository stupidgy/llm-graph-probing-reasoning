#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import networkx as nx
import matplotlib
import pandas as pd
from scipy.stats import pearsonr

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

def print_top_edges(matrix, title, top_n=200, use_absolute=False):
    """打印网络中权重最大的边
    
    参数:
        matrix: 网络矩阵
        title: 标题描述
        top_n: 要打印的边数量
        use_absolute: 是否使用绝对值进行排序
    """
    # 将矩阵转换为边列表格式
    n = matrix.shape[0]
    all_edges = []
    
    # 收集所有边（只考虑下三角，避免重复）
    for i in range(n):
        for j in range(i+1, n):
            weight = matrix[i, j]
            if weight != 0:  # 只收集非零边
                all_edges.append((i, j, abs(weight) if use_absolute else weight, weight))
    
    # 检查是否有边
    if not all_edges:
        print(f"警告: 未找到非零边")
        return
    
    # 按权重排序边（如果use_absolute=True则按绝对值排序，否则按原始值排序）
    sort_key = 2 if use_absolute else 3
    all_edges.sort(key=lambda x: x[sort_key], reverse=True)
    
    # 打印前top_n个边
    print(f"\n{title} - 前{min(top_n, len(all_edges))}个权重最大的边:")
    print(f"{'节点1':<8} {'节点2':<8} {'权重':<12}")
    print("-" * 30)
    
    for i, (node1, node2, _, weight) in enumerate(all_edges[:top_n]):
        print(f"{node1:<8} {node2:<8} {weight:<12.6f}")

def analyze_node_degrees(matrices, titles, output_path, threshold=0.1):
    """分析和可视化不同网络中节点度的对比
    
    参数:
        matrices: 网络矩阵列表
        titles: 对应的标题列表
        output_path: 输出文件路径
        threshold: 考虑边的权重阈值
    """
    if not matrices or len(matrices) != len(titles):
        print("错误: 矩阵和标题数量不匹配")
        return
    
    plt.figure(figsize=(15, 10))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # 不同网络使用不同颜色
    
    # 计算每个网络的节点加权度
    for idx, (matrix, title) in enumerate(zip(matrices, titles)):
        if matrix is None:
            continue
            
        n = matrix.shape[0]
        
        # 计算每个节点的加权度（权重绝对值之和）
        weighted_degrees = {}
        for i in range(n):
            # 计算与该节点相连的所有边的权重绝对值之和
            # 排除对角线元素（自环）
            row_sum = np.sum(np.abs(matrix[i, :][np.arange(n) != i]))
            col_sum = np.sum(np.abs(matrix[:, i][np.arange(n) != i]))
            # 由于是对称矩阵，行和列的和应该相等，但为了避免浮点误差，取平均值
            weighted_degrees[i] = (row_sum + col_sum) / 2
        
        degrees_list = list(weighted_degrees.values())
        
        # 绘制加权度分布直方图
        color = colors[idx % len(colors)]
        sns.histplot(degrees_list, bins=30, kde=True, color=color, alpha=0.7, 
                     label=f"{title} (avg={np.mean(degrees_list):.2f})")
        
        # 打印加权度统计信息
        print(f"\n{title} 节点加权度统计:")
        print(f"  节点数: {n}")
        print(f"  平均加权度: {np.mean(degrees_list):.4f}")
        print(f"  最大加权度: {np.max(degrees_list):.4f} (节点 {max(weighted_degrees, key=weighted_degrees.get)})")
        print(f"  最小加权度: {np.min(degrees_list):.4f} (节点 {min(weighted_degrees, key=weighted_degrees.get)})")
        print(f"  加权度为0的节点数: {sum(1 for d in degrees_list if d == 0)}")
        print(f"  标准差: {np.std(degrees_list):.4f}")
        
        # 找出加权度最大的前10个节点
        top_nodes = sorted(weighted_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"  加权度最大的10个节点:")
        for node, degree in top_nodes:
            print(f"    节点 {node}: 加权度 = {degree:.4f}")
        
        # 保存节点加权度数据到CSV
        df = pd.DataFrame({"Node": list(weighted_degrees.keys()), "Weighted Degree": list(weighted_degrees.values())})
        df.sort_values(by="Weighted Degree", ascending=False, inplace=True)
        df.to_csv(f"{output_path}_{title.replace(' ', '_')}_weighted_degrees.csv", index=False)
    
    plt.title("Node Weighted Degree Distribution Comparison", fontsize=16)
    plt.xlabel("Node Weighted Degree", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(f"{output_path}_weighted_degree_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 创建加权度对比散点图
    if len(matrices) >= 2 and matrices[0] is not None and matrices[1] is not None:
        plt.figure(figsize=(10, 8))
        
        # 确保两个矩阵大小相同
        min_size = min(matrices[0].shape[0], matrices[1].shape[0])
        matrix1 = matrices[0][:min_size, :min_size]
        matrix2 = matrices[1][:min_size, :min_size]
        
        # 计算每个节点在两个网络中的加权度
        weighted_degrees1 = {}
        weighted_degrees2 = {}
        
        for i in range(min_size):
            # 第一个网络的加权度
            row_sum1 = np.sum(np.abs(matrix1[i, :][np.arange(min_size) != i]))
            col_sum1 = np.sum(np.abs(matrix1[:, i][np.arange(min_size) != i]))
            weighted_degrees1[i] = (row_sum1 + col_sum1) / 2
            
            # 第二个网络的加权度
            row_sum2 = np.sum(np.abs(matrix2[i, :][np.arange(min_size) != i]))
            col_sum2 = np.sum(np.abs(matrix2[:, i][np.arange(min_size) != i]))
            weighted_degrees2[i] = (row_sum2 + col_sum2) / 2
        
        # 创建加权度对比数据
        nodes = range(min_size)
        degree_pairs = [(weighted_degrees1.get(n, 0), weighted_degrees2.get(n, 0)) for n in nodes]
        
        # 计算相关系数
        x_vals = [p[0] for p in degree_pairs]
        y_vals = [p[1] for p in degree_pairs]
        correlation, p_value = pearsonr(x_vals, y_vals)
        
        # 绘制散点图
        plt.scatter(x_vals, y_vals, alpha=0.5)
        
        # 添加回归线
        m, b = np.polyfit(x_vals, y_vals, 1)
        plt.plot(x_vals, [m*x + b for x in x_vals], color='red', linestyle='--')
        
        plt.title(f"Node Weighted Degree Comparison: {titles[0]} vs {titles[1]}\nCorrelation: {correlation:.4f}", fontsize=14)
        plt.xlabel(f"{titles[0]} Node Weighted Degree", fontsize=12)
        plt.ylabel(f"{titles[1]} Node Weighted Degree", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 添加45度参考线
        max_degree = max(max(x_vals), max(y_vals))
        plt.plot([0, max_degree], [0, max_degree], color='green', alpha=0.5, linestyle=':')
        
        plt.tight_layout()
        plt.savefig(f"{output_path}_weighted_degree_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 创建加权度差异分布直方图
        plt.figure(figsize=(10, 6))
        degree_diffs = [d2 - d1 for d1, d2 in zip(x_vals, y_vals)]
        sns.histplot(degree_diffs, bins=30, kde=True)
        plt.axvline(x=0, color='red', linestyle='--')
        plt.title(f"Node Weighted Degree Difference: {titles[1]} - {titles[0]}\nAverage Difference: {np.mean(degree_diffs):.4f}", fontsize=14)
        plt.xlabel("Node Weighted Degree Difference", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_path}_weighted_degree_difference.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 打印加权度差异的统计信息
        print(f"\n节点加权度差异统计 ({titles[1]} - {titles[0]}):")
        print(f"  平均差异: {np.mean(degree_diffs):.4f}")
        print(f"  标准差: {np.std(degree_diffs):.4f}")
        print(f"  最大正差异: {np.max(degree_diffs):.4f}")
        print(f"  最大负差异: {np.min(degree_diffs):.4f}")
        
        # 找出加权度差异最大的节点
        diff_with_nodes = [(i, degree_diffs[i]) for i in range(len(degree_diffs))]
        top_diff_pos = sorted(diff_with_nodes, key=lambda x: x[1], reverse=True)[:10]
        top_diff_neg = sorted(diff_with_nodes, key=lambda x: x[1])[:10]
        
        print(f"  加权度差异最大的10个节点 ({titles[1]} > {titles[0]}):")
        for node, diff in top_diff_pos:
            print(f"    节点 {node}: 差异 = {diff:.4f} ({weighted_degrees2.get(node, 0):.4f} - {weighted_degrees1.get(node, 0):.4f})")
        
        print(f"  加权度差异最小的10个节点 ({titles[0]} > {titles[1]}):")
        for node, diff in top_diff_neg:
            print(f"    节点 {node}: 差异 = {diff:.4f} ({weighted_degrees2.get(node, 0):.4f} - {weighted_degrees1.get(node, 0):.4f})")
    
    print(f"节点加权度分析完成，结果保存在: {output_path}")

def compare_edge_weights(matrix1, matrix2, title1, title2, output_path, top_n=1000):
    """对比两个网络中边的权重，选取权重绝对值排前1000的边，取并集后制作散点图
    
    参数:
        matrix1: 第一个网络矩阵
        matrix2: 第二个网络矩阵
        title1: 第一个网络标题
        title2: 第二个网络标题
        output_path: 输出文件路径
        top_n: 每个网络中选取的边数量
    """
    if matrix1 is None or matrix2 is None:
        print("Error: Both matrices must be provided")
        return
    
    # 确保两个矩阵大小相同
    min_size = min(matrix1.shape[0], matrix2.shape[0])
    matrix1 = matrix1[:min_size, :min_size]
    matrix2 = matrix2[:min_size, :min_size]
    
    # 收集两个网络的所有边
    edges1 = {}
    edges2 = {}
    
    # 收集第一个网络的边（只考虑下三角，避免重复）
    for i in range(min_size):
        for j in range(i+1, min_size):
            weight = matrix1[i, j]
            if weight != 0:  # 只收集非零边
                edges1[(i, j)] = weight
    
    # 收集第二个网络的边
    for i in range(min_size):
        for j in range(i+1, min_size):
            weight = matrix2[i, j]
            if weight != 0:  # 只收集非零边
                edges2[(i, j)] = weight
    
    # 按权重绝对值排序
    sorted_edges1 = sorted(edges1.items(), key=lambda x: abs(x[1]), reverse=True)
    sorted_edges2 = sorted(edges2.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # 取前top_n个边
    top_edges1 = sorted_edges1[:top_n] if len(sorted_edges1) > top_n else sorted_edges1
    top_edges2 = sorted_edges2[:top_n] if len(sorted_edges2) > top_n else sorted_edges2
    
    # 提取边的键（节点对）
    top_edges1_keys = set(edge[0] for edge in top_edges1)
    top_edges2_keys = set(edge[0] for edge in top_edges2)
    
    # 取并集
    union_edges = top_edges1_keys.union(top_edges2_keys)
    print(f"Selected {len(union_edges)} unique edges (union of top {top_n} edges from each network)")
    
    # 准备散点图数据
    x_vals = []
    y_vals = []
    
    for edge in union_edges:
        x_val = edges1.get(edge, 0)  # 如果边不存在，权重为0
        y_val = edges2.get(edge, 0)
        x_vals.append(x_val)
        y_vals.append(y_val)
    
    # 计算相关系数
    correlation, p_value = pearsonr(x_vals, y_vals)
    
    # 绘制散点图
    plt.figure(figsize=(12, 10))
    
    # 定义颜色映射函数 - 基于两个网络中权重的差异
    colors = []
    for x, y in zip(x_vals, y_vals):
        # 计算差异
        diff = abs(x) - abs(y)
        # 红色表示第一个网络权重更大，蓝色表示第二个网络权重更大
        if diff > 0:
            colors.append('red')
        elif diff < 0:
            colors.append('blue')
        else:
            colors.append('gray')
    
    # 绘制散点图
    plt.scatter(x_vals, y_vals, alpha=0.7, c=colors)
    
    # 添加回归线
    m, b = np.polyfit(x_vals, y_vals, 1)
    sorted_x = sorted(x_vals)
    plt.plot(sorted_x, [m*x + b for x in sorted_x], color='green', linestyle='--')
    
    # 添加45度参考线
    max_val = max(max(abs(min(x_vals)), abs(max(x_vals))), max(abs(min(y_vals)), abs(max(y_vals))))
    plt.plot([-max_val, max_val], [-max_val, max_val], color='black', alpha=0.5, linestyle=':')
    
    # 添加坐标轴
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    
    # 添加标题和标签
    plt.title(f"Edge Weight Comparison: {title1} vs {title2}\nCorrelation: {correlation:.4f}", fontsize=16)
    plt.xlabel(f"{title1} Edge Weight", fontsize=14)
    plt.ylabel(f"{title2} Edge Weight", fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label=f'Higher in {title1}'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label=f'Higher in {title2}'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Equal')
    ]
    plt.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    plt.savefig(f"{output_path}_edge_weight_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存边权重数据到CSV
    edge_data = []
    for edge, x, y in zip(union_edges, x_vals, y_vals):
        i, j = edge
        edge_data.append({
            "Node1": i,
            "Node2": j,
            f"{title1}_Weight": x,
            f"{title2}_Weight": y,
            "Abs_Difference": abs(x) - abs(y)
        })
    
    df = pd.DataFrame(edge_data)
    df.sort_values(by="Abs_Difference", ascending=False, inplace=True)
    df.to_csv(f"{output_path}_edge_weight_comparison.csv", index=False)
    
    print(f"Edge weight comparison completed. Results saved to: {output_path}_edge_weight_comparison.png")
    
    # 打印一些统计信息
    print(f"\n边权重对比统计:")
    print(f"  总边数: {len(union_edges)}")
    print(f"  相关系数: {correlation:.4f} (p-value: {p_value:.4e})")
    print(f"  {title1}中权重绝对值更大的边: {sum(1 for x, y in zip(x_vals, y_vals) if abs(x) > abs(y))}")
    print(f"  {title2}中权重绝对值更大的边: {sum(1 for x, y in zip(x_vals, y_vals) if abs(x) < abs(y))}")
    print(f"  权重绝对值相等的边: {sum(1 for x, y in zip(x_vals, y_vals) if abs(x) == abs(y))}")
    
    # 打印权重差异最大的前10条边
    print(f"\n权重差异最大的10条边 ({title1} > {title2}):")
    top_diff = sorted(zip(union_edges, x_vals, y_vals), key=lambda t: abs(t[1])-abs(t[2]), reverse=True)[:10]
    for (i, j), x, y in top_diff:
        print(f"  节点 {i}-{j}: {x:.4f} vs {y:.4f}, 差异: {abs(x)-abs(y):.4f}")
    
    print(f"\n权重差异最大的10条边 ({title2} > {title1}):")
    bottom_diff = sorted(zip(union_edges, x_vals, y_vals), key=lambda t: abs(t[1])-abs(t[2]))[:10]
    for (i, j), x, y in bottom_diff:
        print(f"  节点 {i}-{j}: {x:.4f} vs {y:.4f}, 差异: {abs(x)-abs(y):.4f}")

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
    parser.add_argument('--print_top', type=int, default=50,
                        help='Print top N edges by weight (set to 0 to disable)')
    parser.add_argument('--degree_threshold', type=float, default=0,
                        help='Threshold for considering an edge in degree calculation')
    parser.add_argument('--analyze_degrees', action='store_true',
                        help='Perform node degree analysis')
    parser.add_argument('--compare_edge_weights', action='store_true',
                        help='Compare edge weights between Think and NoThink networks')
    parser.add_argument('--top_edges_compare', type=int, default=524288,
                        help='Number of top edges to include in edge weight comparison')
    
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
    
    # 打印前N个权重最大的边
    if args.print_top > 0:
        if difference_network is not None:
            print_top_edges(difference_network, "差异网络(Think - NoThink)", args.print_top, use_absolute=True)
    
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
    
    # 执行节点度分析
    if args.analyze_degrees:
        print("\n执行节点度分析...")
        matrices = [think_network, nothink_network, difference_network]
        titles = ["Think Model", "NoThink Model", "Difference Network"]
        analyze_node_degrees(matrices, titles, os.path.join(args.output_dir, "node_degrees"), 
                            threshold=args.degree_threshold)
    
    # 执行边权重对比分析
    if args.compare_edge_weights and think_network is not None and nothink_network is not None:
        print("\n执行边权重对比分析...")
        compare_edge_weights(
            think_network, 
            nothink_network, 
            "Think Model", 
            "NoThink Model", 
            os.path.join(args.output_dir, "edge_weights"),
            top_n=args.top_edges_compare
        )
    
    print(f"\nVisualization completed. Results saved in {args.output_dir} directory")

if __name__ == "__main__":
    main() 