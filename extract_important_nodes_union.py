#!/usr/bin/env python3
import json
import argparse
from pathlib import Path

def extract_important_nodes_union(degree_file, elements_file, output_file=None):
    """
    提取think和nothink模式下重要节点的并集
    包括：度重要性节点 + 重要节点 + 重要边的端点
    分别统计每个模式，并进行交集差集分析
    """
    
    # 读取度重要性文件
    with open(degree_file, 'r', encoding='utf-8') as f:
        degree_data = json.load(f)
    
    # 读取重要元素文件
    with open(elements_file, 'r', encoding='utf-8') as f:
        elements_data = json.load(f)
    
    print("=== Think模式重要节点统计 ===")
    
    # Think模式 - 度重要性节点
    think_degree_nodes = set(degree_data['think']['top_nodes']['indices'])
    print(f"Think模式度重要性节点数量: {len(think_degree_nodes)}")
    
    # Think模式 - 重要节点
    think_important_nodes = set(elements_data['think']['top_nodes']['indices'])
    print(f"Think模式重要节点数量: {len(think_important_nodes)}")
    
    # Think模式 - 重要边的端点
    think_edge_endpoints = set()
    for connection in elements_data['think']['top_edges']['connections']:
        think_edge_endpoints.add(connection[0])  # 起始端点
        think_edge_endpoints.add(connection[1])  # 结束端点
    print(f"Think模式重要边端点数量: {len(think_edge_endpoints)}")
    
    # Think模式并集
    think_union = set()
    think_union.update(think_degree_nodes)
    think_union.update(think_important_nodes)
    think_union.update(think_edge_endpoints)
    think_union_sorted = sorted(list(think_union))
    
    print(f"Think模式总并集节点数: {len(think_union)}")
    
    print("\n=== NoThink模式重要节点统计 ===")
    
    # NoThink模式 - 度重要性节点
    nothink_degree_nodes = set(degree_data['nothink']['top_nodes']['indices'])
    print(f"NoThink模式度重要性节点数量: {len(nothink_degree_nodes)}")
    
    # NoThink模式 - 重要节点
    nothink_important_nodes = set(elements_data['nothink']['top_nodes']['indices'])
    print(f"NoThink模式重要节点数量: {len(nothink_important_nodes)}")
    
    # NoThink模式 - 重要边的端点
    nothink_edge_endpoints = set()
    for connection in elements_data['nothink']['top_edges']['connections']:
        nothink_edge_endpoints.add(connection[0])  # 起始端点
        nothink_edge_endpoints.add(connection[1])  # 结束端点
    print(f"NoThink模式重要边端点数量: {len(nothink_edge_endpoints)}")
    
    # NoThink模式并集
    nothink_union = set()
    nothink_union.update(nothink_degree_nodes)
    nothink_union.update(nothink_important_nodes)
    nothink_union.update(nothink_edge_endpoints)
    nothink_union_sorted = sorted(list(nothink_union))
    
    print(f"NoThink模式总并集节点数: {len(nothink_union)}")
    
    
    # 转换为逗号分隔的字符串格式
    think_union_str = ",".join(map(str, think_union_sorted))
    nothink_union_str = ",".join(map(str, nothink_union_sorted))

    
    # 保存结果
    result = {
        "description": "Think和NoThink模式下重要节点的分离统计",
        "think_mode": {
            "degree_nodes": sorted(list(think_degree_nodes)),
            "important_nodes": sorted(list(think_important_nodes)),
            "edge_endpoints": sorted(list(think_edge_endpoints)),
            "union_nodes": think_union_sorted,
            "union_nodes_str": think_union_str,
            "union_count": len(think_union)
        },
        "nothink_mode": {
            "degree_nodes": sorted(list(nothink_degree_nodes)),
            "important_nodes": sorted(list(nothink_important_nodes)),
            "edge_endpoints": sorted(list(nothink_edge_endpoints)),
            "union_nodes": nothink_union_sorted,
            "union_nodes_str": nothink_union_str,
            "union_count": len(nothink_union)
        },
    }
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {output_file}")
    
    return result

def main():
    parser = argparse.ArgumentParser(description="提取Think和NoThink模式下重要节点的并集")
    parser.add_argument('--degree_file', 
                       default='importance_analysis_output/node_degree_importance_top_100.json',
                       help='度重要性文件路径')
    parser.add_argument('--elements_file', 
                       default='importance_analysis_output/top_100_important_elements.json',
                       help='重要元素文件路径')
    parser.add_argument('--output', 
                       default='importance_analysis_output/important_nodes_union.json',
                       help='输出文件路径')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not Path(args.degree_file).exists():
        print(f"错误: 度重要性文件不存在: {args.degree_file}")
        return
    
    if not Path(args.elements_file).exists():
        print(f"错误: 重要元素文件不存在: {args.elements_file}")
        return
    
    # 提取重要节点并集
    result = extract_important_nodes_union(args.degree_file, args.elements_file, args.output)

if __name__ == "__main__":
    main() 