#!/usr/bin/env python3
"""
基于现有分析报告的干预范围分析脚本
从各个实验文件夹的analysis_report.md中提取正确率数据，
分析干预是否超出了原始正确率的变化范围
包含总体、按学科、按难度等级的详细分析
"""

import os
import re
from typing import Dict, List, Tuple

def extract_accuracy_from_report(file_path: str) -> Dict:
    """从analysis_report.md文件中提取正确率数据"""
    if not os.path.exists(file_path):
        return {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        result = {}
        
        # 1. 提取总体准确率
        pattern = r'\| intervention_gaussian_replace_mean0_std0 \| ([0-9.%]+) \([0-9/]+\) \| ([0-9.%]+) \([0-9/]+\) \| ([0-9.%]+) \([0-9/]+\) \| ([0-9.%]+) \([0-9/]+\) \|'
        match = re.search(pattern, content)
        
        if match:
            result['overall'] = {
                'NoThink': {
                    'original': float(match.group(1).rstrip('%')),
                    'intervention': float(match.group(2).rstrip('%'))
                },
                'Think': {
                    'original': float(match.group(3).rstrip('%')),
                    'intervention': float(match.group(4).rstrip('%'))
                }
            }
        
        # 2. 提取按学科的准确率 - 修正正则表达式
        subject_pattern = r'\| ([^|]+?) \| \d+ \| ([0-9.%]+) \([0-9/]+\) \| ([0-9.%]+) \([0-9/]+\) \| [^|]+ \| ([0-9.%]+) \([0-9/]+\) \| ([0-9.%]+) \([0-9/]+\) \| [^|]+ \|'
        subjects_section = re.search(r'### intervention_gaussian_replace_mean0_std0 - 按学科准确率.*?(?=###|##|$)', content, re.DOTALL)
        
        if subjects_section:
            result['subjects'] = {}
            subject_matches = list(re.finditer(subject_pattern, subjects_section.group(0)))
            
            for match in subject_matches:
                subject = match.group(1).strip()
                if subject != '学科' and not subject.startswith('---'):  # 跳过表头和分隔线
                    try:
                        result['subjects'][subject] = {
                            'NoThink': {
                                'original': float(match.group(2).rstrip('%')),
                                'intervention': float(match.group(3).rstrip('%'))
                            },
                            'Think': {
                                'original': float(match.group(4).rstrip('%')),
                                'intervention': float(match.group(5).rstrip('%'))
                            }
                        }
                    except:
                        pass  # 跳过无效行
        
        # 3. 提取按难度等级的准确率 - 修正正则表达式
        level_pattern = r'\| Level (\d+) \| \d+ \| ([0-9.%]+) \([0-9/]+\) \| ([0-9.%]+) \([0-9/]+\) \| [^|]+ \| ([0-9.%]+) \([0-9/]+\) \| ([0-9.%]+) \([0-9/]+\) \| [^|]+ \|'
        levels_section = re.search(r'### intervention_gaussian_replace_mean0_std0 - 按难度等级.*?(?=###|##|$)', content, re.DOTALL)
        
        if levels_section:
            result['levels'] = {}
            level_matches = list(re.finditer(level_pattern, levels_section.group(0)))
            
            for match in level_matches:
                level = f"Level {match.group(1)}"
                try:
                    result['levels'][level] = {
                        'NoThink': {
                            'original': float(match.group(2).rstrip('%')),
                            'intervention': float(match.group(3).rstrip('%'))
                        },
                        'Think': {
                            'original': float(match.group(4).rstrip('%')),
                            'intervention': float(match.group(5).rstrip('%'))
                        }
                    }
                except:
                    pass  # 跳过无效行
        
        return result
        
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return {}

def analyze_category_data(all_data: Dict, category: str, category_name: str):
    """分析某个分类的数据（学科或难度等级）"""
    print(f"\n🔍 {category_name}分析:")
    
    # 收集所有该分类的原始正确率
    category_stats = {}
    
    for exp_name, exp_data in all_data.items():
        if category not in exp_data:
            continue
            
        for item_name, item_data in exp_data[category].items():
            if item_name not in category_stats:
                category_stats[item_name] = {
                    'NoThink': {'originals': [], 'interventions': []},
                    'Think': {'originals': [], 'interventions': []}
                }
            
            for mode in ['NoThink', 'Think']:
                if mode in item_data:
                    category_stats[item_name][mode]['originals'].append(item_data[mode]['original'])
                    category_stats[item_name][mode]['interventions'].append(item_data[mode]['intervention'])
    
    # 分析每个类别项
    significant_items = []
    
    print(f"{'类别':<20} {'模式':<8} {'原始范围':<15} {'干预范围':<15} {'超出范围的实验'}")
    print("-" * 80)
    
    for item_name in sorted(category_stats.keys()):
        item_data = category_stats[item_name]
        
        for mode in ['NoThink', 'Think']:
            originals = item_data[mode]['originals']
            interventions = item_data[mode]['interventions']
            
            if not originals:
                continue
                
            orig_min, orig_max = min(originals), max(originals)
            
            # 检查哪些干预超出范围
            outside_experiments = []
            for i, (exp_name, exp_data) in enumerate(all_data.items()):
                if category in exp_data and item_name in exp_data[category]:
                    if mode in exp_data[category][item_name]:
                        inter_val = exp_data[category][item_name][mode]['intervention']
                        if inter_val < orig_min or inter_val > orig_max:
                            short_name = exp_name.replace('math_intervention_results_', '').replace('math_intervention_results', 'baseline')
                            outside_experiments.append(short_name)
            
            outside_str = ", ".join(outside_experiments) if outside_experiments else "无"
            if outside_experiments:
                significant_items.append(f"{item_name}-{mode}")
            
            print(f"{item_name:<20} {mode:<8} {orig_min:.1f}%-{orig_max:.1f}%{'':<3} {min(interventions):.1f}%-{max(interventions):.1f}%{'':<3} {outside_str}")
    
    if significant_items:
        print(f"\n📈 发现 {len(significant_items)} 个{category_name}类别有显著干预效果:")
        for item in significant_items:
            print(f"    • {item}")
    else:
        print(f"\n📈 所有{category_name}类别的干预效果都在原始正确率范围内")

def analyze_experiments():
    """分析所有实验结果"""
    
    # 定义实验目录
    experiment_dirs = [
        'math_intervention_results',
        'math_intervention_results_nothink_1epoch_42nodes',
        'math_intervention_results_nothink_1epoch_42nodes_greedy',
        'math_intervention_results_think_1epoch',
        'math_intervention_results_think_1epoch_43nodes',
        'math_intervention_results_think_1epoch_43nodes_greedy',
        'math_intervention_results_think_bynode_1epoch'
    ]
    
    all_data = {}
    
    # 收集所有实验数据
    for exp_dir in experiment_dirs:
        report_path = os.path.join(exp_dir, 'analysis_report.md')
        data = extract_accuracy_from_report(report_path)
        if data:
            all_data[exp_dir] = data
            print(f"✓ 成功读取 {exp_dir}")
        else:
            print(f"✗ 无法读取 {exp_dir}")
    
    if not all_data:
        print("没有找到任何有效的实验数据！")
        return
    
    # 1. 总体分析
    nothink_originals = []
    think_originals = []
    
    for exp_name, data in all_data.items():
        if 'overall' in data:
            if 'NoThink' in data['overall']:
                nothink_originals.append(data['overall']['NoThink']['original'])
            if 'Think' in data['overall']:
                think_originals.append(data['overall']['Think']['original'])
    
    # 计算原始正确率的范围
    nothink_min = min(nothink_originals) if nothink_originals else 0
    nothink_max = max(nothink_originals) if nothink_originals else 0
    think_min = min(think_originals) if think_originals else 0
    think_max = max(think_originals) if think_originals else 0
    
    print("\n" + "="*80)
    print("干预效果详细范围分析报告")
    print("="*80)
    
    print(f"\n📊 总体正确率范围统计:")
    print(f"  NoThink模式: {nothink_min:.2f}% - {nothink_max:.2f}% (变化范围: {nothink_max - nothink_min:.2f}%)")
    print(f"  Think模式:   {think_min:.2f}% - {think_max:.2f}% (变化范围: {think_max - think_min:.2f}%)")
    
    print(f"\n🔍 总体干预效果分析:")
    print(f"{'实验名称':<45} {'NoThink原始':<12} {'NoThink干预':<12} {'干预效果':<15} {'Think原始':<12} {'Think干预':<12} {'干预效果':<15}")
    print("-" * 140)
    
    significant_interventions = []
    
    for exp_name, data in all_data.items():
        if 'overall' not in data:
            continue
            
        nothink_orig = data['overall'].get('NoThink', {}).get('original', 0)
        nothink_inter = data['overall'].get('NoThink', {}).get('intervention', 0)
        think_orig = data['overall'].get('Think', {}).get('original', 0)
        think_inter = data['overall'].get('Think', {}).get('intervention', 0)
        
        # 判断干预是否超出原始范围
        nothink_outside = nothink_inter < nothink_min or nothink_inter > nothink_max if nothink_inter != 0 else False
        think_outside = think_inter < think_min or think_inter > think_max if think_inter != 0 else False
        
        nothink_effect = "超出范围!" if nothink_outside else "范围内"
        think_effect = "超出范围!" if think_outside else "范围内"
        
        if nothink_outside or think_outside:
            significant_interventions.append(exp_name)
        
        # 简化实验名称显示
        short_name = exp_name.replace('math_intervention_results_', '').replace('math_intervention_results', 'baseline')
        
        print(f"{short_name:<45} {nothink_orig:<12.2f} {nothink_inter:<12.2f} {nothink_effect:<15} {think_orig:<12.2f} {think_inter:<12.2f} {think_effect:<15}")
    
    # 2. 按学科分析
    if any('subjects' in data for data in all_data.values()):
        analyze_category_data(all_data, 'subjects', '学科')
    
    # 3. 按难度等级分析
    if any('levels' in data for data in all_data.values()):
        analyze_category_data(all_data, 'levels', '难度等级')
    
    print("\n" + "="*80)
    print("📈 总体干预效果总结:")
    
    if significant_interventions:
        print(f"  发现 {len(significant_interventions)} 个实验的总体干预效果超出了原始正确率变化范围:")
        for exp in significant_interventions:
            short_name = exp.replace('math_intervention_results_', '').replace('math_intervention_results', 'baseline')
            print(f"    • {short_name}")
    else:
        print("  所有实验的总体干预效果都在原始正确率的变化范围内，表明干预影响可能不显著")
    
    print(f"\n💡 解释:")
    print(f"  - 由于采样随机性，不同实验的原始正确率会有变化")
    print(f"  - 如果干预后正确率超出了原始正确率的最大最小值范围，可能表明干预有显著影响")
    print(f"  - 在范围内的变化可能主要由采样随机性造成")
    print(f"  - 按学科和难度等级的分析能够发现更细粒度的干预效果")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    analyze_experiments() 