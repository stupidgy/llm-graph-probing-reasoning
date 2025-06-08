import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import networkx as nx
from scipy.stats import pearsonr, spearmanr
import random
import matplotlib

# 设置matplotlib字体，解决中文显示问题
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans', 'Bitstream Vera Sans', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置模型路径
MODEL1_PATH = "data/graph_probing/data4/huguangyi/models/Qwen/Qwen3-0.6B"
MODEL2_PATH = "data/graph_probing/data4/huguangyi/models/Qwen/Qwen3-0.6B_nothink"
LAYER = 14
OUTPUT_DIR = "network_analysis_results"
SAMPLE_SIZE = 1855  # 每个模型随机采样的网络数量，减少计算量

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_sample_networks(model_path, layer, sample_size=SAMPLE_SIZE):
    """从模型中加载随机采样的网络"""
    networks = []
    network_ids = []
    
    # 获取所有可能的样本ID
    all_sample_ids = []
    for sample_id in os.listdir(model_path):
        sample_path = os.path.join(model_path, sample_id)
        if os.path.isdir(sample_path):
            network_file = os.path.join(sample_path, f"layer_{layer}_corr.npy")
            if os.path.exists(network_file):
                all_sample_ids.append(sample_id)
    
    # 随机采样
    if len(all_sample_ids) > sample_size:
        sampled_ids = random.sample(all_sample_ids, sample_size)
    else:
        sampled_ids = all_sample_ids
    
    # 加载采样的网络
    for sample_id in tqdm(sampled_ids, desc=f"加载 {os.path.basename(model_path)} 的网络"):
        try:
            sample_path = os.path.join(model_path, sample_id)
            network_file = os.path.join(sample_path, f"layer_{layer}_corr.npy")
            network = np.load(network_file)
            networks.append(network)
            network_ids.append(sample_id)
        except Exception as e:
            print(f"加载样本 {sample_id} 时出错: {e}")
    
    return networks, network_ids

def compute_basic_stats(networks):
    """计算网络的基本统计特性"""
    n_networks = len(networks)
    if n_networks == 0:
        return None
    
    # 检查网络规模
    shapes = [net.shape for net in networks]
    unique_shapes = set(shapes)
    print(f"发现的网络规模: {unique_shapes}")
    
    # 计算基本统计量
    stats = {
        "网络数量": n_networks,
        "平均网络规模": np.mean([s[0] for s in shapes]),
        "最小网络规模": min([s[0] for s in shapes]),
        "最大网络规模": max([s[0] for s in shapes]),
    }
    
    # 检查是否所有网络都是方阵
    all_square = all(s[0] == s[1] for s in shapes)
    stats["全部是方阵"] = all_square
    
    return stats

def analyze_network_sample(networks, threshold=0.2):
    """分析采样网络的详细特性"""
    all_densities = []
    all_mean_weights = []
    all_std_weights = []
    all_max_weights = []
    all_min_weights = []
    all_avg_degrees = []
    all_clustering_coefs = []
    
    for network in tqdm(networks, desc="分析网络特性"):
        # 排除对角线元素
        mask = ~np.eye(network.shape[0], dtype=bool)
        weights = network[mask]
        
        # 计算基本统计量
        all_densities.append(np.count_nonzero(weights) / weights.size)
        all_mean_weights.append(np.mean(weights))
        all_std_weights.append(np.std(weights))
        all_max_weights.append(np.max(weights))
        all_min_weights.append(np.min(weights))
        
        # 将相关矩阵转换为网络图
        G = nx.Graph()
        n = network.shape[0]
        G.add_nodes_from(range(n))
        
        for i in range(n):
            for j in range(i+1, n):
                if abs(network[i, j]) > threshold:
                    G.add_edge(i, j, weight=network[i, j])
        
        # 计算网络特性
        degrees = [d for _, d in G.degree()]
        all_avg_degrees.append(np.mean(degrees))
        
        clustering = nx.clustering(G)
        if clustering:
            avg_clustering = sum(clustering.values()) / len(clustering)
            all_clustering_coefs.append(avg_clustering)
    
    # 汇总结果
    results = {
        "密度": {
            "平均值": np.mean(all_densities),
            "标准差": np.std(all_densities),
            "最小值": np.min(all_densities),
            "最大值": np.max(all_densities),
        },
        "边权重": {
            "平均值": np.mean(all_mean_weights),
            "标准差": np.mean(all_std_weights),
            "最小值": np.mean(all_min_weights),
            "最大值": np.mean(all_max_weights),
        },
        "平均度": {
            "平均值": np.mean(all_avg_degrees),
            "标准差": np.std(all_avg_degrees),
            "最小值": np.min(all_avg_degrees),
            "最大值": np.max(all_avg_degrees),
        },
        "聚类系数": {
            "平均值": np.mean(all_clustering_coefs) if all_clustering_coefs else 0,
            "标准差": np.std(all_clustering_coefs) if all_clustering_coefs else 0,
            "最小值": np.min(all_clustering_coefs) if all_clustering_coefs else 0,
            "最大值": np.max(all_clustering_coefs) if all_clustering_coefs else 0,
        }
    }
    
    return results

def plot_weight_distribution_sample(networks, model_name):
    """绘制边权重分布（从采样的网络中）"""
    # 随机选择一个网络进行分析
    if not networks:
        print(f"错误：{model_name} 没有可用的网络")
        return
    
    sample_network = random.choice(networks)
    mask = ~np.eye(sample_network.shape[0], dtype=bool)
    weights = sample_network[mask]
    
    plt.figure(figsize=(10, 6))
    sns.histplot(weights, kde=True)
    plt.title(f"{model_name} 边权重分布（样本）")
    plt.xlabel("边权重")
    plt.ylabel("频率")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_weight_distribution_sample.png"), dpi=300)
    plt.close()
    
    # 分析网络中的连接强度分布
    positive_weights = weights[weights > 0]
    negative_weights = weights[weights < 0]
    
    stats = {
        "正权重比例": len(positive_weights) / len(weights) if len(weights) > 0 else 0,
        "负权重比例": len(negative_weights) / len(weights) if len(weights) > 0 else 0,
        "正权重平均值": np.mean(positive_weights) if len(positive_weights) > 0 else 0,
        "负权重平均值": np.mean(negative_weights) if len(negative_weights) > 0 else 0,
    }
    
    return stats

def analyze_network_similarity_sample(networks1, networks2, network_ids1, network_ids2):
    """分析两组网络的相似性（使用ID匹配）"""
    # 查找两个模型中共同的网络ID
    common_ids = set(network_ids1).intersection(set(network_ids2))
    if not common_ids:
        print("错误：两个模型没有共同的网络ID")
        return None
    
    pearson_correlations = []
    spearman_correlations = []
    common_id_list = list(common_ids)
    
    for id in tqdm(common_id_list, desc="分析网络相似性"):
        idx1 = network_ids1.index(id)
        idx2 = network_ids2.index(id)
        
        net1 = networks1[idx1]
        net2 = networks2[idx2]
        
        # 确保两个网络大小相同
        min_size = min(net1.shape[0], net2.shape[0])
        net1 = net1[:min_size, :min_size]
        net2 = net2[:min_size, :min_size]
        
        # 将矩阵转换为向量（排除对角线）
        mask = ~np.eye(min_size, dtype=bool)
        vec1 = net1[mask]
        vec2 = net2[mask]
        
        # 计算相关系数
        pearson_corr, _ = pearsonr(vec1, vec2)
        spearman_corr, _ = spearmanr(vec1, vec2)
        
        pearson_correlations.append(pearson_corr)
        spearman_correlations.append(spearman_corr)
    
    results = {
        "Pearson相关系数": {
            "平均值": np.mean(pearson_correlations),
            "标准差": np.std(pearson_correlations),
            "最小值": np.min(pearson_correlations),
            "最大值": np.max(pearson_correlations),
        },
        "Spearman相关系数": {
            "平均值": np.mean(spearman_correlations),
            "标准差": np.std(spearman_correlations),
            "最小值": np.min(spearman_correlations),
            "最大值": np.max(spearman_correlations),
        }
    }
    
    # 绘制相关系数分布
    plt.figure(figsize=(10, 6))
    sns.histplot(pearson_correlations, kde=True)
    plt.axvline(np.mean(pearson_correlations), color='r', linestyle='--', 
                label=f'平均值: {np.mean(pearson_correlations):.4f}')
    plt.title("两个模型网络的Pearson相关系数分布")
    plt.xlabel("Pearson相关系数")
    plt.ylabel("频率")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "pearson_correlation_distribution.png"), dpi=300)
    plt.close()
    
    return results

def compute_average_network(networks):
    """计算多个网络的平均网络矩阵"""
    if not networks:
        return None
    
    # 检查所有网络的尺寸是否相同
    first_shape = networks[0].shape
    for net in networks[1:]:
        if net.shape != first_shape:
            print(f"警告: 发现不同尺寸的网络。第一个网络: {first_shape}, 当前网络: {net.shape}")
    
    # 找到最小的网络尺寸，以确保所有网络可以对齐
    min_shape = min([net.shape[0] for net in networks])
    
    # 裁剪所有网络到相同尺寸并累加
    aligned_networks = [net[:min_shape, :min_shape] for net in networks]
    average_network = np.mean(aligned_networks, axis=0)
    
    return average_network

def plot_network_heatmap(network, title, output_path):
    """将网络矩阵绘制为热度图"""
    # 打印网络矩阵的统计信息以进行诊断
    print(f"\n热度图 '{title}' 的数据统计:")
    print(f"  最大值: {np.max(network):.6f}")
    print(f"  最小值: {np.min(network):.6f}")
    print(f"  平均值: {np.mean(network):.6f}")
    print(f"  标准差: {np.std(network):.6f}")
    print(f"  中位数: {np.median(network):.6f}")
    print(f"  非零值占比: {np.count_nonzero(network)/network.size*100:.2f}%")
    
    # 如果数据范围太小，调整对比度
    vmax = max(abs(np.max(network)), abs(np.min(network)))
    if vmax < 0.01:  # 如果最大值太小
        print(f"  警告: 数据范围过小 ({vmax:.6f})，将进行对比度增强")
        # 对网络进行归一化处理以增强对比度
        network_enhanced = network.copy()
        # 使用百分比阈值进行截断，保留极端值
        p_low, p_high = np.percentile(network_enhanced, [5, 95])
        print(f"  5%分位数: {p_low:.6f}, 95%分位数: {p_high:.6f}")
        vmax = max(abs(p_low), abs(p_high))
    else:
        network_enhanced = network.copy()
    
    # 创建下三角矩阵热度图
    plt.figure(figsize=(12, 10))
    mask = np.zeros_like(network, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True  # 只显示下三角矩阵，避免重复
    
    # 使用英文标题避免字体问题
    eng_title = title
    if "Think模型" in title:
        eng_title = f"Think Model Average Correlation Network (Layer {LAYER})"
    elif "NoThink模型" in title:
        eng_title = f"NoThink Model Average Correlation Network (Layer {LAYER})"
    elif "差异网络" in title:
        eng_title = f"Difference Network between Think and NoThink Models (Layer {LAYER})"
    
    # 创建热度图，使用vmax和vmin设置颜色范围
    sns.heatmap(network_enhanced, mask=mask, cmap="coolwarm", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.5},
                vmin=-vmax, vmax=vmax)  # 使用对称的颜色范围
    
    plt.title(eng_title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    # 生成带颜色条的完整热度图
    plt.figure(figsize=(14, 12))
    sns.heatmap(network_enhanced, cmap="coolwarm", center=0,
                square=True, linewidths=0.5, vmin=-vmax, vmax=vmax)
    
    plt.title(eng_title + " (Full)", fontsize=16) 
    plt.tight_layout()
    plt.savefig(output_path.replace('.png', '_full.png'), dpi=300)
    plt.close()
    
    # 额外生成一个使用更极端颜色映射的版本
    plt.figure(figsize=(14, 12))
    # 使用不同的色彩映射和更强的对比度
    sns.heatmap(network_enhanced, cmap="RdBu_r", center=0,
                square=True, linewidths=0.5, 
                robust=True,  # 使用鲁棒统计量而不是极值
                annot=False)  # 不显示数值标注以避免过于拥挤
    
    plt.title(eng_title + " (Enhanced Contrast)", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path.replace('.png', '_enhanced.png'), dpi=300)
    plt.close()
    
    # 保存数据分布直方图以帮助分析
    plt.figure(figsize=(10, 6))
    sns.histplot(network.flatten(), bins=50, kde=True)
    plt.title(f"Value Distribution - {eng_title}", fontsize=14)
    plt.xlabel("Correlation Value")
    plt.ylabel("Frequency")
    plt.savefig(output_path.replace('.png', '_distribution.png'), dpi=300)
    plt.close()

def main():
    # 设置随机种子以确保可重复性
    random.seed(42)
    
    # 加载两个模型的网络样本
    print("加载网络数据...")
    networks1, network_ids1 = load_sample_networks(MODEL1_PATH, LAYER)
    networks2, network_ids2 = load_sample_networks(MODEL2_PATH, LAYER)
    
    # 计算基本统计量
    print("\n计算基本统计量...")
    stats1 = compute_basic_stats(networks1)
    stats2 = compute_basic_stats(networks2)
    
    print(f"\n模型1 ({os.path.basename(MODEL1_PATH)}) 基本统计量:")
    for key, value in stats1.items():
        print(f"{key}: {value}")
    
    print(f"\n模型2 ({os.path.basename(MODEL2_PATH)}) 基本统计量:")
    for key, value in stats2.items():
        print(f"{key}: {value}")
    
    # 计算平均网络并绘制热度图
    print("\n计算平均网络并绘制热度图...")
    avg_network1 = compute_average_network(networks1)
    avg_network2 = compute_average_network(networks2)
    
    if avg_network1 is not None:
        print(f"模型1平均网络形状: {avg_network1.shape}")
        plot_network_heatmap(
            avg_network1, 
            f"Think模型平均相关网络 (层 {LAYER})", 
            os.path.join(OUTPUT_DIR, "think_average_network_heatmap.png")
        )
    
    if avg_network2 is not None:
        print(f"模型2平均网络形状: {avg_network2.shape}")
        plot_network_heatmap(
            avg_network2, 
            f"NoThink模型平均相关网络 (层 {LAYER})", 
            os.path.join(OUTPUT_DIR, "nothink_average_network_heatmap.png")
        )
    
    # 计算差异网络并绘制热度图
    if avg_network1 is not None and avg_network2 is not None:
        diff_network = avg_network1 - avg_network2
        print(f"差异网络形状: {diff_network.shape}")
        plot_network_heatmap(
            diff_network, 
            f"Think与NoThink模型的差异网络 (层 {LAYER})", 
            os.path.join(OUTPUT_DIR, "difference_network_heatmap.png")
        )
        
        # 保存平均网络和差异网络
        np.save(os.path.join(OUTPUT_DIR, "think_average_network.npy"), avg_network1)
        np.save(os.path.join(OUTPUT_DIR, "nothink_average_network.npy"), avg_network2)
        np.save(os.path.join(OUTPUT_DIR, "difference_network.npy"), diff_network)
    
    # 分析网络特性
    print("\n分析网络特性...")
    results1 = analyze_network_sample(networks1)
    results2 = analyze_network_sample(networks2)
    
    # 绘制边权重分布
    print("\n分析边权重分布...")
    weight_stats1 = plot_weight_distribution_sample(networks1, os.path.basename(MODEL1_PATH))
    weight_stats2 = plot_weight_distribution_sample(networks2, os.path.basename(MODEL2_PATH))
    
    # 分析网络相似性
    print("\n分析网络相似性...")
    similarity_results = analyze_network_similarity_sample(networks1, networks2, network_ids1, network_ids2)
    
    # 打印结果
    print("\n=============== 网络特性统计结果 ===============")
    print(f"\n模型1 ({os.path.basename(MODEL1_PATH)}):")
    for category, stats in results1.items():
        print(f"\n{category}:")
        for stat_name, value in stats.items():
            print(f"  {stat_name}: {value:.4f}")
    
    print(f"\n模型2 ({os.path.basename(MODEL2_PATH)}):")
    for category, stats in results2.items():
        print(f"\n{category}:")
        for stat_name, value in stats.items():
            print(f"  {stat_name}: {value:.4f}")
    
    print("\n=============== 边权重分布分析 ===============")
    print(f"\n模型1 ({os.path.basename(MODEL1_PATH)}):")
    for stat_name, value in weight_stats1.items():
        print(f"  {stat_name}: {value:.4f}")
    
    print(f"\n模型2 ({os.path.basename(MODEL2_PATH)}):")
    for stat_name, value in weight_stats2.items():
        print(f"  {stat_name}: {value:.4f}")
    
    if similarity_results:
        print("\n=============== 网络相似性分析 ===============")
        for category, stats in similarity_results.items():
            print(f"\n{category}:")
            for stat_name, value in stats.items():
                print(f"  {stat_name}: {value:.4f}")
    
    # 保存结果到CSV - 使用新的方式构建DataFrame
    data = []
    for category in results1.keys():
        for stat_name in results1[category].keys():
            data.append({
                "特性": category,
                "统计量": stat_name,
                "模型1": results1[category][stat_name],
                "模型2": results2[category][stat_name]
            })
    
    results_df = pd.DataFrame(data)
    
    # 保存结果
    results_df.to_csv(os.path.join(OUTPUT_DIR, "network_analysis_results.csv"), index=False)
    
    print(f"\n分析完成。结果保存在 {OUTPUT_DIR} 目录。")

if __name__ == "__main__":
    main() 