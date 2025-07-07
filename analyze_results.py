#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from typing import List, Dict
import glob

def load_latest_chunk_files(results_dir: str) -> List[Dict]:
    """加载最新的chunk文件数据"""
    
    # 查找所有temp文件
    temp_files = glob.glob(os.path.join(results_dir, "chunk_*_temp_*.json"))
    
    # 按chunk分组
    chunk_files = {}
    for file in temp_files:
        filename = os.path.basename(file)
        # 提取chunk号和数据数量，格式如：chunk_0_temp_240.json
        parts = filename.replace('.json', '').split('_')
        chunk_id = int(parts[1])
        data_count = int(parts[3])
        
        if chunk_id not in chunk_files or data_count > chunk_files[chunk_id][1]:
            chunk_files[chunk_id] = (file, data_count)
    
    print("找到的最新chunk文件:")
    for chunk_id, (file, count) in chunk_files.items():
        print(f"  Chunk {chunk_id}: {os.path.basename(file)} ({count}个数据)")
    
    # 加载所有最新chunk的数据
    all_data = []
    for chunk_id, (file, count) in sorted(chunk_files.items()):
        print(f"\n正在加载 {os.path.basename(file)}...")
        try:
            with open(file, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
                all_data.extend(chunk_data)
                print(f"  成功加载 {len(chunk_data)} 个问题")
        except Exception as e:
            print(f"  加载失败: {e}")
    
    # 按problem_index排序
    all_data.sort(key=lambda x: x.get('problem_index', 0))
    
    return all_data

def calculate_statistics(results: List[Dict]) -> Dict:
    """计算统计信息"""
    
    # 收集所有实验配置
    all_configs = set()
    for result in results:
        for exp_name in result['experiments'].keys():
            all_configs.add(exp_name)
    
    all_configs = sorted(list(all_configs))
    
    stats = {}
    
    for config_name in all_configs:
        # 统计该配置下的准确率
        nothink_orig_correct = 0
        nothink_interv_correct = 0
        think_orig_correct = 0
        think_interv_correct = 0
        total_nothink = 0
        total_think = 0
        
        # 新增：交集分析
        nothink_correct_to_wrong = 0  # 原来正确，干预后错误
        nothink_wrong_to_correct = 0  # 原来错误，干预后正确
        nothink_both_correct = 0      # 干预前后都正确
        nothink_both_wrong = 0        # 干预前后都错误
        
        think_correct_to_wrong = 0
        think_wrong_to_correct = 0
        think_both_correct = 0
        think_both_wrong = 0
        
        for result in results:
            if config_name in result['experiments']:
                exp_result = result['experiments'][config_name]
                
                # 检查是否有错误
                if 'error' in exp_result:
                    continue
                
                # NoThink模式分析
                if 'nothink_mode' in exp_result:
                    total_nothink += 1
                    orig_correct = exp_result['nothink_mode']['original_correct']
                    interv_correct = exp_result['nothink_mode']['intervention_correct']
                    
                    if orig_correct:
                        nothink_orig_correct += 1
                    if interv_correct:
                        nothink_interv_correct += 1
                    
                    # 交集分析
                    if orig_correct and interv_correct:
                        nothink_both_correct += 1
                    elif orig_correct and not interv_correct:
                        nothink_correct_to_wrong += 1
                    elif not orig_correct and interv_correct:
                        nothink_wrong_to_correct += 1
                    else:  # not orig_correct and not interv_correct
                        nothink_both_wrong += 1
                
                # Think模式分析
                if 'think_mode' in exp_result:
                    total_think += 1
                    orig_correct = exp_result['think_mode']['original_correct']
                    interv_correct = exp_result['think_mode']['intervention_correct']
                    
                    if orig_correct:
                        think_orig_correct += 1
                    if interv_correct:
                        think_interv_correct += 1
                    
                    # 交集分析
                    if orig_correct and interv_correct:
                        think_both_correct += 1
                    elif orig_correct and not interv_correct:
                        think_correct_to_wrong += 1
                    elif not orig_correct and interv_correct:
                        think_wrong_to_correct += 1
                    else:  # not orig_correct and not interv_correct
                        think_both_wrong += 1
        
        nothink_orig_acc = (nothink_orig_correct / total_nothink * 100) if total_nothink > 0 else 0
        nothink_interv_acc = (nothink_interv_correct / total_nothink * 100) if total_nothink > 0 else 0
        think_orig_acc = (think_orig_correct / total_think * 100) if total_think > 0 else 0
        think_interv_acc = (think_interv_correct / total_think * 100) if total_think > 0 else 0
        
        stats[config_name] = {
            'nothink': {
                'original': {'correct': nothink_orig_correct, 'total': total_nothink, 'accuracy': nothink_orig_acc},
                'intervention': {'correct': nothink_interv_correct, 'total': total_nothink, 'accuracy': nothink_interv_acc},
                'transition': {
                    'correct_to_wrong': nothink_correct_to_wrong,
                    'wrong_to_correct': nothink_wrong_to_correct,
                    'both_correct': nothink_both_correct,
                    'both_wrong': nothink_both_wrong
                }
            },
            'think': {
                'original': {'correct': think_orig_correct, 'total': total_think, 'accuracy': think_orig_acc},
                'intervention': {'correct': think_interv_correct, 'total': total_think, 'accuracy': think_interv_acc},
                'transition': {
                    'correct_to_wrong': think_correct_to_wrong,
                    'wrong_to_correct': think_wrong_to_correct,
                    'both_correct': think_both_correct,
                    'both_wrong': think_both_wrong
                }
            }
        }
    
    return stats

def analyze_by_subject(results: List[Dict]) -> Dict:
    """按学科分析统计"""
    subject_stats = {}
    
    for result in results:
        subject = result.get('subject', 'Unknown')
        level = result.get('level', 'Unknown')
        
        if subject not in subject_stats:
            subject_stats[subject] = {'total': 0, 'levels': {}}
        
        subject_stats[subject]['total'] += 1
        
        if level not in subject_stats[subject]['levels']:
            subject_stats[subject]['levels'][level] = 0
        subject_stats[subject]['levels'][level] += 1
    
    return subject_stats

def analyze_by_subject_accuracy(results: List[Dict]) -> Dict:
    """按学科分析准确率统计"""
    subject_accuracy_stats = {}
    
    # 收集所有实验配置
    all_configs = set()
    for result in results:
        for exp_name in result['experiments'].keys():
            all_configs.add(exp_name)
    
    all_configs = sorted(list(all_configs))
    
    for config_name in all_configs:
        subject_accuracy_stats[config_name] = {}
        
        # 按subject分组统计
        subject_data = {}
        
        for result in results:
            subject = result.get('subject', 'Unknown')
            if subject not in subject_data:
                subject_data[subject] = {
                    'nothink_orig_correct': 0, 'nothink_interv_correct': 0,
                    'think_orig_correct': 0, 'think_interv_correct': 0,
                    'total_nothink': 0, 'total_think': 0,
                    # 添加交集分析
                    'nothink_correct_to_wrong': 0, 'nothink_wrong_to_correct': 0,
                    'nothink_both_correct': 0, 'nothink_both_wrong': 0,
                    'think_correct_to_wrong': 0, 'think_wrong_to_correct': 0,
                    'think_both_correct': 0, 'think_both_wrong': 0
                }
            
            if config_name in result['experiments']:
                exp_result = result['experiments'][config_name]
                
                # 检查是否有错误
                if 'error' in exp_result:
                    continue
                
                if 'nothink_mode' in exp_result:
                    subject_data[subject]['total_nothink'] += 1
                    orig_correct = exp_result['nothink_mode']['original_correct']
                    interv_correct = exp_result['nothink_mode']['intervention_correct']
                    
                    if orig_correct:
                        subject_data[subject]['nothink_orig_correct'] += 1
                    if interv_correct:
                        subject_data[subject]['nothink_interv_correct'] += 1
                    
                    # 交集分析
                    if orig_correct and interv_correct:
                        subject_data[subject]['nothink_both_correct'] += 1
                    elif orig_correct and not interv_correct:
                        subject_data[subject]['nothink_correct_to_wrong'] += 1
                    elif not orig_correct and interv_correct:
                        subject_data[subject]['nothink_wrong_to_correct'] += 1
                    else:
                        subject_data[subject]['nothink_both_wrong'] += 1
                
                if 'think_mode' in exp_result:
                    subject_data[subject]['total_think'] += 1
                    orig_correct = exp_result['think_mode']['original_correct']
                    interv_correct = exp_result['think_mode']['intervention_correct']
                    
                    if orig_correct:
                        subject_data[subject]['think_orig_correct'] += 1
                    if interv_correct:
                        subject_data[subject]['think_interv_correct'] += 1
                    
                    # 交集分析
                    if orig_correct and interv_correct:
                        subject_data[subject]['think_both_correct'] += 1
                    elif orig_correct and not interv_correct:
                        subject_data[subject]['think_correct_to_wrong'] += 1
                    elif not orig_correct and interv_correct:
                        subject_data[subject]['think_wrong_to_correct'] += 1
                    else:
                        subject_data[subject]['think_both_wrong'] += 1
        
        # 计算每个subject的准确率
        for subject, data in subject_data.items():
            nothink_orig_acc = (data['nothink_orig_correct'] / data['total_nothink'] * 100) if data['total_nothink'] > 0 else 0
            nothink_interv_acc = (data['nothink_interv_correct'] / data['total_nothink'] * 100) if data['total_nothink'] > 0 else 0
            think_orig_acc = (data['think_orig_correct'] / data['total_think'] * 100) if data['total_think'] > 0 else 0
            think_interv_acc = (data['think_interv_correct'] / data['total_think'] * 100) if data['total_think'] > 0 else 0
            
            subject_accuracy_stats[config_name][subject] = {
                'nothink': {
                    'original': {'correct': data['nothink_orig_correct'], 'total': data['total_nothink'], 'accuracy': nothink_orig_acc},
                    'intervention': {'correct': data['nothink_interv_correct'], 'total': data['total_nothink'], 'accuracy': nothink_interv_acc},
                    'transition': {
                        'correct_to_wrong': data['nothink_correct_to_wrong'],
                        'wrong_to_correct': data['nothink_wrong_to_correct'],
                        'both_correct': data['nothink_both_correct'],
                        'both_wrong': data['nothink_both_wrong']
                    }
                },
                'think': {
                    'original': {'correct': data['think_orig_correct'], 'total': data['total_think'], 'accuracy': think_orig_acc},
                    'intervention': {'correct': data['think_interv_correct'], 'total': data['total_think'], 'accuracy': think_interv_acc},
                    'transition': {
                        'correct_to_wrong': data['think_correct_to_wrong'],
                        'wrong_to_correct': data['think_wrong_to_correct'],
                        'both_correct': data['think_both_correct'],
                        'both_wrong': data['think_both_wrong']
                    }
                }
            }
    
    return subject_accuracy_stats

def analyze_by_level(results: List[Dict]) -> Dict:
    """按难度等级分析统计"""
    level_stats = {}
    
    # 收集所有实验配置
    all_configs = set()
    for result in results:
        for exp_name in result['experiments'].keys():
            all_configs.add(exp_name)
    
    all_configs = sorted(list(all_configs))
    
    for config_name in all_configs:
        level_stats[config_name] = {}
        
        # 按level分组统计
        level_data = {}
        
        for result in results:
            level = result.get('level', 'Unknown')
            if level not in level_data:
                level_data[level] = {
                    'nothink_orig_correct': 0, 'nothink_interv_correct': 0,
                    'think_orig_correct': 0, 'think_interv_correct': 0,
                    'total_nothink': 0, 'total_think': 0,
                    # 添加完整的交集分析
                    'nothink_correct_to_wrong': 0, 'nothink_wrong_to_correct': 0,
                    'nothink_both_correct': 0, 'nothink_both_wrong': 0,
                    'think_correct_to_wrong': 0, 'think_wrong_to_correct': 0,
                    'think_both_correct': 0, 'think_both_wrong': 0
                }
            
            if config_name in result['experiments']:
                exp_result = result['experiments'][config_name]
                
                # 检查是否有错误
                if 'error' in exp_result:
                    continue
                
                if 'nothink_mode' in exp_result:
                    level_data[level]['total_nothink'] += 1
                    orig_correct = exp_result['nothink_mode']['original_correct']
                    interv_correct = exp_result['nothink_mode']['intervention_correct']
                    
                    if orig_correct:
                        level_data[level]['nothink_orig_correct'] += 1
                    if interv_correct:
                        level_data[level]['nothink_interv_correct'] += 1
                    
                    # 完整的交集分析
                    if orig_correct and interv_correct:
                        level_data[level]['nothink_both_correct'] += 1
                    elif orig_correct and not interv_correct:
                        level_data[level]['nothink_correct_to_wrong'] += 1
                    elif not orig_correct and interv_correct:
                        level_data[level]['nothink_wrong_to_correct'] += 1
                    else:  # not orig_correct and not interv_correct
                        level_data[level]['nothink_both_wrong'] += 1
                
                if 'think_mode' in exp_result:
                    level_data[level]['total_think'] += 1
                    orig_correct = exp_result['think_mode']['original_correct']
                    interv_correct = exp_result['think_mode']['intervention_correct']
                    
                    if orig_correct:
                        level_data[level]['think_orig_correct'] += 1
                    if interv_correct:
                        level_data[level]['think_interv_correct'] += 1
                    
                    # 完整的交集分析
                    if orig_correct and interv_correct:
                        level_data[level]['think_both_correct'] += 1
                    elif orig_correct and not interv_correct:
                        level_data[level]['think_correct_to_wrong'] += 1
                    elif not orig_correct and interv_correct:
                        level_data[level]['think_wrong_to_correct'] += 1
                    else:  # not orig_correct and not interv_correct
                        level_data[level]['think_both_wrong'] += 1
        
        # 计算每个level的准确率
        for level, data in level_data.items():
            nothink_orig_acc = (data['nothink_orig_correct'] / data['total_nothink'] * 100) if data['total_nothink'] > 0 else 0
            nothink_interv_acc = (data['nothink_interv_correct'] / data['total_nothink'] * 100) if data['total_nothink'] > 0 else 0
            think_orig_acc = (data['think_orig_correct'] / data['total_think'] * 100) if data['total_think'] > 0 else 0
            think_interv_acc = (data['think_interv_correct'] / data['total_think'] * 100) if data['total_think'] > 0 else 0
            
            level_stats[config_name][level] = {
                'nothink': {
                    'original': {'correct': data['nothink_orig_correct'], 'total': data['total_nothink'], 'accuracy': nothink_orig_acc},
                    'intervention': {'correct': data['nothink_interv_correct'], 'total': data['total_nothink'], 'accuracy': nothink_interv_acc},
                    'transition': {
                        'correct_to_wrong': data['nothink_correct_to_wrong'],
                        'wrong_to_correct': data['nothink_wrong_to_correct'],
                        'both_correct': data['nothink_both_correct'],
                        'both_wrong': data['nothink_both_wrong']
                    }
                },
                'think': {
                    'original': {'correct': data['think_orig_correct'], 'total': data['total_think'], 'accuracy': think_orig_acc},
                    'intervention': {'correct': data['think_interv_correct'], 'total': data['total_think'], 'accuracy': think_interv_acc},
                    'transition': {
                        'correct_to_wrong': data['think_correct_to_wrong'],
                        'wrong_to_correct': data['think_wrong_to_correct'],
                        'both_correct': data['think_both_correct'],
                        'both_wrong': data['think_both_wrong']
                    }
                }
            }
    
    return level_stats

def generate_report(stats: Dict, subject_stats: Dict, subject_accuracy_stats: Dict, level_stats: Dict, total_problems: int, output_file: str):
    """生成统计报告"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# MATH数据集神经干预实验统计报告\n\n")
        f.write(f"## 实验概览\n\n")
        f.write(f"- **总问题数**: {total_problems}\n")
        f.write(f"- **实验配置数**: {len(stats)}\n\n")
        
        # 按学科统计
        f.write("## 按学科统计\n\n")
        f.write("| 学科 | 问题数 | 占比 | 难度分布 |\n")
        f.write("|------|--------|------|---------|\n")
        
        for subject, data in sorted(subject_stats.items(), key=lambda x: x[1]['total'], reverse=True):
            percentage = data['total'] / total_problems * 100
            level_dist = ', '.join([f"Level {k}: {v}" for k, v in sorted(data['levels'].items())])
            f.write(f"| {subject} | {data['total']} | {percentage:.1f}% | {level_dist} |\n")
        
        f.write("\n")
        
        # 总体准确率统计
        f.write("## 总体准确率统计\n\n")
        f.write("| 实验配置 | NoThink原始 | NoThink干预 | Think原始 | Think干预 | 变化趋势 |\n")
        f.write("|----------|-------------|-------------|-----------|-----------|----------|\n")
        
        for config_name, config_stats in stats.items():
            nothink_orig = config_stats['nothink']['original']
            nothink_interv = config_stats['nothink']['intervention']
            think_orig = config_stats['think']['original']
            think_interv = config_stats['think']['intervention']
            
            # 计算变化
            nothink_change = nothink_interv['accuracy'] - nothink_orig['accuracy']
            think_change = think_interv['accuracy'] - think_orig['accuracy']
            
            nothink_trend = "↑" if nothink_change > 0 else "↓" if nothink_change < 0 else "→"
            think_trend = "↑" if think_change > 0 else "↓" if think_change < 0 else "→"
            
            f.write(f"| {config_name} | "
                   f"{nothink_orig['accuracy']:.1f}% ({nothink_orig['correct']}/{nothink_orig['total']}) | "
                   f"{nothink_interv['accuracy']:.1f}% ({nothink_interv['correct']}/{nothink_interv['total']}) | "
                   f"{think_orig['accuracy']:.1f}% ({think_orig['correct']}/{think_orig['total']}) | "
                   f"{think_interv['accuracy']:.1f}% ({think_interv['correct']}/{think_interv['total']}) | "
                   f"NT{nothink_trend}{nothink_change:+.1f}%, T{think_trend}{think_change:+.1f}% |\n")
        
        f.write("\n")
        
        # 干预效果交集分析
        f.write("## 干预效果交集分析\n\n")
        f.write("分析干预前后答案正确性的变化情况：\n\n")
        
        for config_name, config_stats in stats.items():
            f.write(f"### {config_name} - 干预效果详细分析\n\n")
            
            nothink_trans = config_stats['nothink']['transition']
            think_trans = config_stats['think']['transition']
            
            f.write("#### NoThink模式交集分析\n\n")
            f.write("| 变化类型 | 问题数 | 占比 | 描述 |\n")
            f.write("|----------|--------|------|----- |\n")
            
            total_nothink = config_stats['nothink']['original']['total']
            
            f.write(f"| 保持正确 | {nothink_trans['both_correct']} | {nothink_trans['both_correct']/total_nothink*100:.1f}% | 干预前后都正确 |\n")
            f.write(f"| 保持错误 | {nothink_trans['both_wrong']} | {nothink_trans['both_wrong']/total_nothink*100:.1f}% | 干预前后都错误 |\n")
            f.write(f"| 正确→错误 | {nothink_trans['correct_to_wrong']} | {nothink_trans['correct_to_wrong']/total_nothink*100:.1f}% | 原来正确，干预后错误 |\n")
            f.write(f"| 错误→正确 | {nothink_trans['wrong_to_correct']} | {nothink_trans['wrong_to_correct']/total_nothink*100:.1f}% | 原来错误，干预后正确 |\n")
            f.write(f"| **总计** | **{total_nothink}** | **100.0%** | 所有问题 |\n")
            
            # 计算净改善
            net_improvement_nothink = nothink_trans['wrong_to_correct'] - nothink_trans['correct_to_wrong']
            improvement_rate_nothink = net_improvement_nothink / total_nothink * 100
            
            f.write(f"\n**NoThink净改善**: {net_improvement_nothink:+d} 个问题 ({improvement_rate_nothink:+.1f}%)\n\n")
            
            f.write("#### Think模式交集分析\n\n")
            f.write("| 变化类型 | 问题数 | 占比 | 描述 |\n")
            f.write("|----------|--------|------|----- |\n")
            
            total_think = config_stats['think']['original']['total']
            
            f.write(f"| 保持正确 | {think_trans['both_correct']} | {think_trans['both_correct']/total_think*100:.1f}% | 干预前后都正确 |\n")
            f.write(f"| 保持错误 | {think_trans['both_wrong']} | {think_trans['both_wrong']/total_think*100:.1f}% | 干预前后都错误 |\n")
            f.write(f"| 正确→错误 | {think_trans['correct_to_wrong']} | {think_trans['correct_to_wrong']/total_think*100:.1f}% | 原来正确，干预后错误 |\n")
            f.write(f"| 错误→正确 | {think_trans['wrong_to_correct']} | {think_trans['wrong_to_correct']/total_think*100:.1f}% | 原来错误，干预后正确 |\n")
            f.write(f"| **总计** | **{total_think}** | **100.0%** | 所有问题 |\n")
            
            # 计算净改善
            net_improvement_think = think_trans['wrong_to_correct'] - think_trans['correct_to_wrong']
            improvement_rate_think = net_improvement_think / total_think * 100
            
            f.write(f"\n**Think净改善**: {net_improvement_think:+d} 个问题 ({improvement_rate_think:+.1f}%)\n\n")
            
            # 模式对比
            f.write("#### 模式对比总结\n\n")
            f.write("| 指标 | NoThink模式 | Think模式 | 差异 |\n")
            f.write("|------|-------------|-----------|------|\n")
            f.write(f"| 干预受益问题数 | {nothink_trans['wrong_to_correct']} | {think_trans['wrong_to_correct']} | {think_trans['wrong_to_correct'] - nothink_trans['wrong_to_correct']:+d} |\n")
            f.write(f"| 干预损害问题数 | {nothink_trans['correct_to_wrong']} | {think_trans['correct_to_wrong']} | {think_trans['correct_to_wrong'] - nothink_trans['correct_to_wrong']:+d} |\n")
            f.write(f"| 净改善问题数 | {net_improvement_nothink:+d} | {net_improvement_think:+d} | {net_improvement_think - net_improvement_nothink:+d} |\n")
            f.write(f"| 净改善率 | {improvement_rate_nothink:+.1f}% | {improvement_rate_think:+.1f}% | {improvement_rate_think - improvement_rate_nothink:+.1f}% |\n")
            
            f.write("\n")
        
        # 按学科准确率分析
        f.write("## 按学科准确率分析\n\n")
        
        for config_name, config_subject_stats in subject_accuracy_stats.items():
            f.write(f"### {config_name} - 按学科准确率\n\n")
            
            f.write("| 学科 | 问题数 | NoThink原始 | NoThink干预 | NoThink变化 | Think原始 | Think干预 | Think变化 |\n")
            f.write("|------|--------|-------------|-------------|-------------|-----------|-----------|----------|\n")
            
            # 按问题数量排序（从多到少）
            sorted_subjects = sorted(config_subject_stats.items(), key=lambda x: x[1]['nothink']['original']['total'], reverse=True)
            
            for subject, subject_data in sorted_subjects:
                nothink_orig = subject_data['nothink']['original']
                nothink_interv = subject_data['nothink']['intervention']
                think_orig = subject_data['think']['original']
                think_interv = subject_data['think']['intervention']
                
                nothink_change = nothink_interv['accuracy'] - nothink_orig['accuracy']
                think_change = think_interv['accuracy'] - think_orig['accuracy']
                
                nothink_trend = "↑" if nothink_change > 0 else "↓" if nothink_change < 0 else "→"
                think_trend = "↑" if think_change > 0 else "↓" if think_change < 0 else "→"
                
                total_problems_subject = nothink_orig['total']
                
                f.write(f"| {subject} | {total_problems_subject} | "
                       f"{nothink_orig['accuracy']:.1f}% ({nothink_orig['correct']}/{nothink_orig['total']}) | "
                       f"{nothink_interv['accuracy']:.1f}% ({nothink_interv['correct']}/{nothink_interv['total']}) | "
                       f"{nothink_trend}{nothink_change:+.1f}% | "
                       f"{think_orig['accuracy']:.1f}% ({think_orig['correct']}/{think_orig['total']}) | "
                       f"{think_interv['accuracy']:.1f}% ({think_interv['correct']}/{think_interv['total']}) | "
                       f"{think_trend}{think_change:+.1f}% |\n")
            
            f.write("\n")
            
            # 添加按学科的详细转换分析
            f.write(f"#### {config_name} - 按学科转换分析\n\n")
            f.write("**NoThink模式转换统计:**\n\n")
            f.write("| 学科 | 保持正确 | 保持错误 | 正确→错误 | 错误→正确 | 净改善 | 净改善率 |\n")
            f.write("|------|----------|----------|----------|----------|--------|---------|\n")
            
            for subject, subject_data in sorted_subjects:
                nothink_trans = subject_data['nothink']['transition']
                total = subject_data['nothink']['original']['total']
                net_improvement = nothink_trans['wrong_to_correct'] - nothink_trans['correct_to_wrong']
                net_rate = (net_improvement / total * 100) if total > 0 else 0
                
                f.write(f"| {subject} | {nothink_trans['both_correct']} | {nothink_trans['both_wrong']} | "
                       f"{nothink_trans['correct_to_wrong']} | {nothink_trans['wrong_to_correct']} | "
                       f"{net_improvement:+d} | {net_rate:+.1f}% |\n")
            
            f.write("\n**Think模式转换统计:**\n\n")
            f.write("| 学科 | 保持正确 | 保持错误 | 正确→错误 | 错误→正确 | 净改善 | 净改善率 |\n")
            f.write("|------|----------|----------|----------|----------|--------|---------|\n")
            
            for subject, subject_data in sorted_subjects:
                think_trans = subject_data['think']['transition']
                total = subject_data['think']['original']['total']
                net_improvement = think_trans['wrong_to_correct'] - think_trans['correct_to_wrong']
                net_rate = (net_improvement / total * 100) if total > 0 else 0
                
                f.write(f"| {subject} | {think_trans['both_correct']} | {think_trans['both_wrong']} | "
                       f"{think_trans['correct_to_wrong']} | {think_trans['wrong_to_correct']} | "
                       f"{net_improvement:+d} | {net_rate:+.1f}% |\n")
            
            f.write("\n")
        
        # 按难度等级分析
        f.write("## 按难度等级分析\n\n")
        
        for config_name, config_level_stats in level_stats.items():
            f.write(f"### {config_name} - 按难度等级\n\n")
            
            f.write("| 难度等级 | 问题数 | NoThink原始 | NoThink干预 | NoThink变化 | Think原始 | Think干预 | Think变化 |\n")
            f.write("|----------|--------|-------------|-------------|-------------|-----------|-----------|----------|\n")
            
            for level in sorted(config_level_stats.keys()):
                level_data = config_level_stats[level]
                nothink_orig = level_data['nothink']['original']
                nothink_interv = level_data['nothink']['intervention']
                think_orig = level_data['think']['original']
                think_interv = level_data['think']['intervention']
                
                nothink_change = nothink_interv['accuracy'] - nothink_orig['accuracy']
                think_change = think_interv['accuracy'] - think_orig['accuracy']
                
                nothink_trend = "↑" if nothink_change > 0 else "↓" if nothink_change < 0 else "→"
                think_trend = "↑" if think_change > 0 else "↓" if think_change < 0 else "→"
                
                total_problems_level = nothink_orig['total']  # 假设nothink和think的总数相同
                
                f.write(f"| Level {level} | {total_problems_level} | "
                       f"{nothink_orig['accuracy']:.1f}% ({nothink_orig['correct']}/{nothink_orig['total']}) | "
                       f"{nothink_interv['accuracy']:.1f}% ({nothink_interv['correct']}/{nothink_interv['total']}) | "
                       f"{nothink_trend}{nothink_change:+.1f}% | "
                       f"{think_orig['accuracy']:.1f}% ({think_orig['correct']}/{think_orig['total']}) | "
                       f"{think_interv['accuracy']:.1f}% ({think_interv['correct']}/{think_interv['total']}) | "
                       f"{think_trend}{think_change:+.1f}% |\n")
            
            f.write("\n")
            
            # 添加按难度等级的详细转换分析
            f.write(f"#### {config_name} - 按难度等级转换分析\n\n")
            f.write("**NoThink模式转换统计:**\n\n")
            f.write("| 难度等级 | 保持正确 | 保持错误 | 正确→错误 | 错误→正确 | 净改善 | 净改善率 |\n")
            f.write("|----------|----------|----------|----------|----------|--------|---------|\n")
            
            for level in sorted(config_level_stats.keys()):
                level_data = config_level_stats[level]
                nothink_trans = level_data['nothink']['transition']
                total = level_data['nothink']['original']['total']
                net_improvement = nothink_trans['wrong_to_correct'] - nothink_trans['correct_to_wrong']
                net_rate = (net_improvement / total * 100) if total > 0 else 0
                
                f.write(f"| Level {level} | {nothink_trans['both_correct']} | {nothink_trans['both_wrong']} | "
                       f"{nothink_trans['correct_to_wrong']} | {nothink_trans['wrong_to_correct']} | "
                       f"{net_improvement:+d} | {net_rate:+.1f}% |\n")
            
            f.write("\n**Think模式转换统计:**\n\n")
            f.write("| 难度等级 | 保持正确 | 保持错误 | 正确→错误 | 错误→正确 | 净改善 | 净改善率 |\n")
            f.write("|----------|----------|----------|----------|----------|--------|---------|\n")
            
            for level in sorted(config_level_stats.keys()):
                level_data = config_level_stats[level]
                think_trans = level_data['think']['transition']
                total = level_data['think']['original']['total']
                net_improvement = think_trans['wrong_to_correct'] - think_trans['correct_to_wrong']
                net_rate = (net_improvement / total * 100) if total > 0 else 0
                
                f.write(f"| Level {level} | {think_trans['both_correct']} | {think_trans['both_wrong']} | "
                       f"{think_trans['correct_to_wrong']} | {think_trans['wrong_to_correct']} | "
                       f"{net_improvement:+d} | {net_rate:+.1f}% |\n")
            
            f.write("\n")
        
        # 详细分析
        f.write("## 详细分析\n\n")
        
        for config_name, config_stats in stats.items():
            f.write(f"### {config_name}\n\n")
            
            nothink_orig = config_stats['nothink']['original']
            nothink_interv = config_stats['nothink']['intervention']
            think_orig = config_stats['think']['original']
            think_interv = config_stats['think']['intervention']
            
            f.write(f"**NoThink模式:**\n")
            f.write(f"- 原始正确率: {nothink_orig['accuracy']:.2f}% ({nothink_orig['correct']}/{nothink_orig['total']})\n")
            f.write(f"- 干预正确率: {nothink_interv['accuracy']:.2f}% ({nothink_interv['correct']}/{nothink_interv['total']})\n")
            f.write(f"- 变化: {nothink_interv['accuracy'] - nothink_orig['accuracy']:+.2f}%\n\n")
            
            f.write(f"**Think模式:**\n")
            f.write(f"- 原始正确率: {think_orig['accuracy']:.2f}% ({think_orig['correct']}/{think_orig['total']})\n")
            f.write(f"- 干预正确率: {think_interv['accuracy']:.2f}% ({think_interv['correct']}/{think_interv['total']})\n")
            f.write(f"- 变化: {think_interv['accuracy'] - think_orig['accuracy']:+.2f}%\n\n")
            
            f.write("---\n\n")

def main():
    """主函数"""
    results_dir = "math_intervention_results_think_1epoch_43nodes_greedy"
    
    if not os.path.exists(results_dir):
        print(f"错误：找不到结果目录 {results_dir}")
        return
    
    print("=== MATH数据集神经干预实验结果统计 ===\n")
    
    # 加载数据
    print("1. 加载实验数据...")
    results = load_latest_chunk_files(results_dir)
    
    if not results:
        print("错误：没有找到有效的实验数据")
        return
    
    print(f"\n总共加载了 {len(results)} 个问题的实验数据")
    
    # 计算统计信息
    print("\n2. 计算统计信息...")
    stats = calculate_statistics(results)
    subject_stats = analyze_by_subject(results)
    subject_accuracy_stats = analyze_by_subject_accuracy(results)
    level_stats = analyze_by_level(results)
    
    # 生成报告
    print("\n3. 生成统计报告...")
    report_file = os.path.join(results_dir, "analysis_report.md")
    generate_report(stats, subject_stats, subject_accuracy_stats, level_stats, len(results), report_file)
    
    # 打印简要统计
    print(f"\n=== 简要统计结果 ===")
    
    for config_name, config_stats in stats.items():
        print(f"\n配置: {config_name}")
        
        nothink_orig = config_stats['nothink']['original']
        nothink_interv = config_stats['nothink']['intervention']
        think_orig = config_stats['think']['original']
        think_interv = config_stats['think']['intervention']
        
        nothink_trans = config_stats['nothink']['transition']
        think_trans = config_stats['think']['transition']
        
        print(f"  NoThink模式: {nothink_orig['accuracy']:.1f}% → {nothink_interv['accuracy']:.1f}% "
              f"({nothink_interv['accuracy'] - nothink_orig['accuracy']:+.1f}%)")
        print(f"  Think模式:   {think_orig['accuracy']:.1f}% → {think_interv['accuracy']:.1f}% "
              f"({think_interv['accuracy'] - think_orig['accuracy']:+.1f}%)")
        
        # 添加交集分析简要信息
        print(f"  交集分析:")
        print(f"    NoThink: 正确→错误({nothink_trans['correct_to_wrong']}) vs 错误→正确({nothink_trans['wrong_to_correct']}) = 净改善({nothink_trans['wrong_to_correct'] - nothink_trans['correct_to_wrong']:+d})")
        print(f"    Think:   正确→错误({think_trans['correct_to_wrong']}) vs 错误→正确({think_trans['wrong_to_correct']}) = 净改善({think_trans['wrong_to_correct'] - think_trans['correct_to_wrong']:+d})")
    
    # 打印按学科的简要统计
    print(f"\n=== 按学科统计 ===")
    
    for config_name, config_subject_stats in subject_accuracy_stats.items():
        print(f"\n配置: {config_name}")
        print("学科                  | NoThink变化 | Think变化 | NT净改善 | T净改善 | 问题数")
        print("---------------------|-------------|-----------|----------|---------|-------")
        
        # 按问题数量排序
        sorted_subjects = sorted(config_subject_stats.items(), key=lambda x: x[1]['nothink']['original']['total'], reverse=True)
        
        for subject, subject_data in sorted_subjects:
            nothink_change = subject_data['nothink']['intervention']['accuracy'] - subject_data['nothink']['original']['accuracy']
            think_change = subject_data['think']['intervention']['accuracy'] - subject_data['think']['original']['accuracy']
            
            # 计算净改善
            nothink_net = subject_data['nothink']['transition']['wrong_to_correct'] - subject_data['nothink']['transition']['correct_to_wrong']
            think_net = subject_data['think']['transition']['wrong_to_correct'] - subject_data['think']['transition']['correct_to_wrong']
            
            total_problems_subject = subject_data['nothink']['original']['total']
            
            print(f"{subject:20} | {nothink_change:+6.1f}%     | {think_change:+6.1f}%   | {nothink_net:+4d}     | {think_net:+4d}    | {total_problems_subject:4d}")
        
        # 添加按学科的详细转换统计
        print(f"\n=== {config_name} - 按学科详细转换分析 ===")
        print("\nNoThink模式转换:")
        print("学科                  | 保持正确 | 保持错误 | 正确→错误 | 错误→正确 | 净改善 | 净改善率")
        print("---------------------|----------|----------|----------|----------|--------|----------")
        
        for subject, subject_data in sorted_subjects:
            nothink_trans = subject_data['nothink']['transition']
            total = subject_data['nothink']['original']['total']
            net_improvement = nothink_trans['wrong_to_correct'] - nothink_trans['correct_to_wrong']
            net_rate = (net_improvement / total * 100) if total > 0 else 0
            
            print(f"{subject:20} | {nothink_trans['both_correct']:8d} | {nothink_trans['both_wrong']:8d} | "
                  f"{nothink_trans['correct_to_wrong']:8d} | {nothink_trans['wrong_to_correct']:8d} | "
                  f"{net_improvement:+6d} | {net_rate:+7.1f}%")
        
        print(f"\nThink模式转换:")
        print("学科                  | 保持正确 | 保持错误 | 正确→错误 | 错误→正确 | 净改善 | 净改善率")
        print("---------------------|----------|----------|----------|----------|--------|----------")
        
        for subject, subject_data in sorted_subjects:
            think_trans = subject_data['think']['transition']
            total = subject_data['think']['original']['total']
            net_improvement = think_trans['wrong_to_correct'] - think_trans['correct_to_wrong']
            net_rate = (net_improvement / total * 100) if total > 0 else 0
            
            print(f"{subject:20} | {think_trans['both_correct']:8d} | {think_trans['both_wrong']:8d} | "
                  f"{think_trans['correct_to_wrong']:8d} | {think_trans['wrong_to_correct']:8d} | "
                  f"{net_improvement:+6d} | {net_rate:+7.1f}%")
    
    # 打印按难度等级的简要统计
    print(f"\n=== 按难度等级统计 ===")
    
    for config_name, config_level_stats in level_stats.items():
        print(f"\n配置: {config_name}")
        print("难度等级 | NoThink变化 | Think变化 | 问题数")
        print("---------|-------------|-----------|-------")
        
        for level in sorted(config_level_stats.keys()):
            level_data = config_level_stats[level]
            nothink_change = level_data['nothink']['intervention']['accuracy'] - level_data['nothink']['original']['accuracy']
            think_change = level_data['think']['intervention']['accuracy'] - level_data['think']['original']['accuracy']
            total_problems_level = level_data['nothink']['original']['total']
            
            print(f"Level {level:2} | {nothink_change:+6.1f}%     | {think_change:+6.1f}%   | {total_problems_level:4d}")
        
        # 添加按难度等级的详细转换统计
        print(f"\n=== {config_name} - 按难度等级详细转换分析 ===")
        print("\nNoThink模式转换:")
        print("难度等级 | 保持正确 | 保持错误 | 正确→错误 | 错误→正确 | 净改善 | 净改善率")
        print("---------|----------|----------|----------|----------|--------|----------")
        
        for level in sorted(config_level_stats.keys()):
            level_data = config_level_stats[level]
            nothink_trans = level_data['nothink']['transition']
            total = level_data['nothink']['original']['total']
            net_improvement = nothink_trans['wrong_to_correct'] - nothink_trans['correct_to_wrong']
            net_rate = (net_improvement / total * 100) if total > 0 else 0
            
            print(f"Level {level:2} | {nothink_trans['both_correct']:8d} | {nothink_trans['both_wrong']:8d} | "
                  f"{nothink_trans['correct_to_wrong']:8d} | {nothink_trans['wrong_to_correct']:8d} | "
                  f"{net_improvement:+6d} | {net_rate:+7.1f}%")
        
        print(f"\nThink模式转换:")
        print("难度等级 | 保持正确 | 保持错误 | 正确→错误 | 错误→正确 | 净改善 | 净改善率")
        print("---------|----------|----------|----------|----------|--------|----------")
        
        for level in sorted(config_level_stats.keys()):
            level_data = config_level_stats[level]
            think_trans = level_data['think']['transition']
            total = level_data['think']['original']['total']
            net_improvement = think_trans['wrong_to_correct'] - think_trans['correct_to_wrong']
            net_rate = (net_improvement / total * 100) if total > 0 else 0
            
            print(f"Level {level:2} | {think_trans['both_correct']:8d} | {think_trans['both_wrong']:8d} | "
                  f"{think_trans['correct_to_wrong']:8d} | {think_trans['wrong_to_correct']:8d} | "
                  f"{net_improvement:+6d} | {net_rate:+7.1f}%")
    
    print(f"\n详细报告已保存到: {report_file}")

if __name__ == "__main__":
    main() 