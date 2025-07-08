#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
import torch
import argparse
import random
from typing import List, Dict, Tuple
from tqdm import tqdm
import copy
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
from collections import defaultdict
import pandas as pd
import pyarrow.parquet as pq
import glob
sys.path.append('.')
sys.path.append('./llm-graph-probing')

# 添加sympy相关导入
import sympy
from sympy import parse_expr
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
transformations = (standard_transformations + (implicit_multiplication_application,))

# 导入现有的神经干预控制器
from neural_intervention import NeuralInterventionController


class MathInterventionExperiment:
    """数学问题神经干预实验类"""
    
    def __init__(self, model_path: str, device: str = "cuda", gpu_id: int = 0):
        """初始化实验控制器"""
        self.controller = NeuralInterventionController(model_path, device, gpu_id)
        self.results = []
    
    def load_openr1_math_dataset(self, dataset_dir: str, split: str = "default", 
                                filter_geometry: bool = True, max_samples: int = None,
                                single_file_index: int = None) -> List[Dict]:
        """
        加载OpenR1-Math数据集
        
        Args:
            dataset_dir: 数据集目录路径 (如 /data4/huguangyi/datasets/OpenR1-Math)
            split: 数据集分割类型 ("default", "extended", "all")
            filter_geometry: 是否只保留几何类型的问题
            max_samples: 最大样本数（用于测试）
            single_file_index: 只加载指定索引的单个文件（0-9，None表示加载所有文件）
            
        Returns:
            加载的数据列表
        """
        print(f"正在加载OpenR1-Math数据集: {dataset_dir}")
        print(f"数据集分割: {split}")
        print(f"几何筛选: {filter_geometry}")
        if single_file_index is not None:
            print(f"单文件模式: 只加载索引 {single_file_index} 的文件")
        
        # 根据分割类型确定数据路径
        if split == "default":
            data_path = os.path.join(dataset_dir, "data")
        elif split == "extended":
            data_path = os.path.join(dataset_dir, "extended")
        elif split == "all":
            data_path = os.path.join(dataset_dir, "all")
        else:
            raise ValueError(f"不支持的数据集分割: {split}")
        
        # 查找所有parquet文件
        parquet_files = glob.glob(os.path.join(data_path, "*.parquet"))
        parquet_files.sort()  # 确保顺序一致
        
        if not parquet_files:
            raise ValueError(f"在 {data_path} 中没有找到parquet文件")
        
        # 如果指定了单文件索引，只加载该文件
        if single_file_index is not None:
            if single_file_index < 0 or single_file_index >= len(parquet_files):
                raise ValueError(f"单文件索引 {single_file_index} 超出范围 [0, {len(parquet_files)-1}]")
            parquet_files = [parquet_files[single_file_index]]
            print(f"单文件模式: 加载文件 {os.path.basename(parquet_files[0])}")
        else:
            print(f"找到 {len(parquet_files)} 个parquet文件")
        
        all_data = []
        geometry_count = 0
        total_count = 0
        
        for parquet_file in parquet_files:
            print(f"正在处理: {os.path.basename(parquet_file)}")
            
            # 读取parquet文件
            table = pq.read_table(parquet_file)
            df = table.to_pandas()
            
            for _, row in df.iterrows():
                total_count += 1
                
                # 如果需要筛选几何题目
                if filter_geometry and row['problem_type'] != 'Geometry':
                    continue
                
                geometry_count += 1
                
                # 转换为统一格式
                item = {
                    'problem': row['problem'],
                    'answer': row['answer'],
                    'solution': row['solution'],
                    'problem_type': row['problem_type'],
                    'question_type': row['question_type'],
                    'source': row['source'],
                    'uuid': row['uuid']
                }
                
                all_data.append(item)
                
                # 检查是否达到最大样本数
                if max_samples and len(all_data) >= max_samples:
                    break
            
            # 检查是否达到最大样本数
            if max_samples and len(all_data) >= max_samples:
                break
        
        print(f"数据加载完成:")
        print(f"- 总处理样本数: {total_count}")
        if filter_geometry:
            print(f"- 几何题目数量: {geometry_count}")
            print(f"- 几何题目占比: {geometry_count/total_count*100:.1f}%")
        print(f"- 最终返回样本数: {len(all_data)}")
        
        return all_data
    
    def load_math_dataset(self, dataset_path: str, max_samples: int = None, 
                         single_file_index: int = None) -> List[Dict]:
        """加载MATH数据集（保持向后兼容）"""
        print(f"正在加载数据集: {dataset_path}")
        
        # 检查是否是OpenR1-Math格式的路径
        if 'OpenR1-Math' in dataset_path:
            # 如果是目录路径，使用新的加载方法
            if os.path.isdir(dataset_path):
                return self.load_openr1_math_dataset(dataset_path, max_samples=max_samples, 
                                                   single_file_index=single_file_index)
            # 如果指定了具体的分割和筛选选项，解析路径
            else:
                # 假设路径格式为: /path/to/OpenR1-Math|split|geometry
                parts = dataset_path.split('|')
                base_path = parts[0]
                split = parts[1] if len(parts) > 1 else "default"
                filter_geometry = parts[2].lower() == 'geometry' if len(parts) > 2 else True
                return self.load_openr1_math_dataset(base_path, split, filter_geometry, max_samples,
                                                   single_file_index)
        
        # 原始MATH数据集加载逻辑
        data = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                if line.strip():
                    item = json.loads(line)
                    data.append(item)
        
        print(f"成功加载 {len(data)} 个数学问题")
        return data
    
    def stratified_sample_openr1_dataset(self, dataset_dir: str, split: str = "default",
                                       filter_geometry: bool = True, target_samples: int = 100, 
                                       random_seed: int = 42, single_file_index: int = None) -> List[Dict]:
        """
        对OpenR1-Math数据集进行分层抽样
        
        Args:
            dataset_dir: 数据集目录路径
            split: 数据集分割类型
            filter_geometry: 是否只保留几何类型的问题
            target_samples: 目标样本数量
            random_seed: 随机种子
            single_file_index: 只加载指定索引的单个文件
            
        Returns:
            分层抽样后的数据集
        """
        # 设置随机种子
        random.seed(random_seed)
        
        print(f"正在对OpenR1-Math数据集进行分层抽样")
        
        # 加载全部数据
        full_dataset = self.load_openr1_math_dataset(dataset_dir, split, filter_geometry, 
                                                   single_file_index=single_file_index)
        
        # 按source和question_type分组
        strata = defaultdict(list)
        strata_counts = defaultdict(int)
        
        for item in full_dataset:
            source = item.get('source', 'Unknown')
            question_type = item.get('question_type', 'Unknown')
            key = f"{source}_{question_type}"
            strata[key].append(item)
            strata_counts[key] += 1
        
        print(f"\n数据集分层统计:")
        print("=" * 60)
        print(f"{'来源_题目类型':<30} {'数量':<8} {'占比':<8}")
        print("-" * 60)
        
        total_items = len(full_dataset)
        for key, count in sorted(strata_counts.items()):
            percentage = count / total_items * 100
            print(f"{key:<30} {count:<8} {percentage:>6.1f}%")
        
        print("-" * 60)
        print(f"{'总计':<30} {total_items:<8} {'100.0%':<8}")
        print("=" * 60)
        
        # 计算每个层级应该抽取的样本数（按比例分配）
        sampled_data = []
        allocation_info = []
        
        for key, items in strata.items():
            # 按比例计算应抽取的样本数
            proportion = len(items) / total_items
            target_for_stratum = max(1, round(target_samples * proportion))  # 至少抽取1个
            
            # 确保不超过该层级的总数
            actual_sample_size = min(target_for_stratum, len(items))
            
            # 随机抽样
            sampled_items = random.sample(items, actual_sample_size)
            sampled_data.extend(sampled_items)
            
            allocation_info.append({
                'stratum': key,
                'total': len(items),
                'target': target_for_stratum,
                'actual': actual_sample_size,
                'proportion': proportion
            })
        
        # 如果抽样数量不足，从较大的层级补充
        if len(sampled_data) < target_samples:
            remaining_needed = target_samples - len(sampled_data)
            print(f"\n需要补充 {remaining_needed} 个样本...")
            
            # 按层级大小排序，从大层级补充
            large_strata = sorted(strata.items(), key=lambda x: len(x[1]), reverse=True)
            
            for key, items in large_strata:
                if remaining_needed <= 0:
                    break
                
                # 获取该层级已抽取的样本
                already_sampled = [item for item in sampled_data 
                                 if (item.get('source', 'Unknown') + '_' + 
                                     item.get('question_type', 'Unknown')) == key]
                
                # 获取未抽取的样本
                remaining_items = [item for item in items if item not in already_sampled]
                
                if remaining_items:
                    additional_samples = min(remaining_needed, len(remaining_items))
                    additional_items = random.sample(remaining_items, additional_samples)
                    sampled_data.extend(additional_items)
                    remaining_needed -= additional_samples
                    
                    # 更新分配信息
                    for info in allocation_info:
                        if info['stratum'] == key:
                            info['actual'] += additional_samples
                            break
        
        # 如果抽样数量过多，随机移除一些
        if len(sampled_data) > target_samples:
            sampled_data = random.sample(sampled_data, target_samples)
        
        print(f"\n分层抽样结果:")
        print("=" * 80)
        print(f"{'来源_题目类型':<30} {'原始数量':<8} {'目标数量':<8} {'实际数量':<8} {'抽样率':<8}")
        print("-" * 80)
        
        # 重新统计最终抽样结果
        final_counts = defaultdict(int)
        for item in sampled_data:
            source = item.get('source', 'Unknown')
            question_type = item.get('question_type', 'Unknown')
            key = f"{source}_{question_type}"
            final_counts[key] += 1
        
        for info in allocation_info:
            key = info['stratum']
            final_actual = final_counts.get(key, 0)
            sampling_rate = final_actual / info['total'] * 100
            print(f"{key:<30} {info['total']:<8} {info['target']:<8} {final_actual:<8} {sampling_rate:>6.1f}%")
        
        print("-" * 80)
        print(f"{'总计':<30} {total_items:<8} {target_samples:<8} {len(sampled_data):<8} {len(sampled_data)/total_items*100:>6.1f}%")
        print("=" * 80)
        
        file_info = f"单文件 {single_file_index}" if single_file_index is not None else "所有文件"
        print(f"\n分层抽样完成！")
        print(f"从 {file_info} 的 {len(full_dataset)} 个几何问题中抽取了 {len(sampled_data)} 个代表性问题")
        
        return sampled_data

    def stratified_sample_dataset(self, dataset_path: str, target_samples: int = 100, 
                                 random_seed: int = 42, single_file_index: int = None) -> List[Dict]:
        """
        分层抽样方法（兼容原有接口）
        """
        # 检查是否是OpenR1-Math格式
        if 'OpenR1-Math' in dataset_path:
            if os.path.isdir(dataset_path):
                return self.stratified_sample_openr1_dataset(dataset_path, target_samples=target_samples, 
                                                          random_seed=random_seed, 
                                                          single_file_index=single_file_index)
            else:
                # 解析路径格式
                parts = dataset_path.split('|')
                base_path = parts[0]
                split = parts[1] if len(parts) > 1 else "default"
                filter_geometry = parts[2].lower() == 'geometry' if len(parts) > 2 else True
                return self.stratified_sample_openr1_dataset(base_path, split, filter_geometry,
                                                          target_samples, random_seed, single_file_index)
        
        # 原有的MATH数据集分层抽样逻辑（不支持单文件功能）
        # 设置随机种子
        random.seed(random_seed)
        
        print(f"正在加载数据集进行分层抽样: {dataset_path}")
        
        # 加载全部数据
        full_dataset = self.load_math_dataset(dataset_path)
        
        # 按level和subject分组
        strata = defaultdict(list)
        level_subject_counts = defaultdict(int)
        
        for item in full_dataset:
            level = item.get('level', 'Unknown')
            subject = item.get('subject', 'Unknown')
            key = f"{level}_{subject}"
            strata[key].append(item)
            level_subject_counts[key] += 1
        
        print(f"\n数据集分层统计:")
        print("=" * 60)
        print(f"{'层级_学科':<25} {'数量':<8} {'占比':<8}")
        print("-" * 60)
        
        total_items = len(full_dataset)
        for key, count in sorted(level_subject_counts.items()):
            percentage = count / total_items * 100
            print(f"{key:<25} {count:<8} {percentage:>6.1f}%")
        
        print("-" * 60)
        print(f"{'总计':<25} {total_items:<8} {'100.0%':<8}")
        print("=" * 60)
        
        # 计算每个层级应该抽取的样本数（按比例分配）
        sampled_data = []
        allocation_info = []
        
        for key, items in strata.items():
            # 按比例计算应抽取的样本数
            proportion = len(items) / total_items
            target_for_stratum = max(1, round(target_samples * proportion))  # 至少抽取1个
            
            # 确保不超过该层级的总数
            actual_sample_size = min(target_for_stratum, len(items))
            
            # 随机抽样
            sampled_items = random.sample(items, actual_sample_size)
            sampled_data.extend(sampled_items)
            
            allocation_info.append({
                'stratum': key,
                'total': len(items),
                'target': target_for_stratum,
                'actual': actual_sample_size,
                'proportion': proportion
            })
        
        # 如果抽样数量不足，从较大的层级补充
        if len(sampled_data) < target_samples:
            remaining_needed = target_samples - len(sampled_data)
            print(f"\n需要补充 {remaining_needed} 个样本...")
            
            # 按层级大小排序，从大层级补充
            large_strata = sorted(strata.items(), key=lambda x: len(x[1]), reverse=True)
            
            for key, items in large_strata:
                if remaining_needed <= 0:
                    break
                
                # 获取该层级已抽取的样本
                already_sampled = [item for item in sampled_data 
                                 if item.get('level', 'Unknown') + '_' + item.get('subject', 'Unknown') == key]
                
                # 获取未抽取的样本
                remaining_items = [item for item in items if item not in already_sampled]
                
                if remaining_items:
                    additional_samples = min(remaining_needed, len(remaining_items))
                    additional_items = random.sample(remaining_items, additional_samples)
                    sampled_data.extend(additional_items)
                    remaining_needed -= additional_samples
                    
                    # 更新分配信息
                    for info in allocation_info:
                        if info['stratum'] == key:
                            info['actual'] += additional_samples
                            break
        
        # 如果抽样数量过多，随机移除一些
        if len(sampled_data) > target_samples:
            sampled_data = random.sample(sampled_data, target_samples)
        
        print(f"\n分层抽样结果:")
        print("=" * 80)
        print(f"{'层级_学科':<25} {'原始数量':<8} {'目标数量':<8} {'实际数量':<8} {'抽样率':<8}")
        print("-" * 80)
        
        # 重新统计最终抽样结果
        final_counts = defaultdict(int)
        for item in sampled_data:
            level = item.get('level', 'Unknown')
            subject = item.get('subject', 'Unknown')
            key = f"{level}_{subject}"
            final_counts[key] += 1
        
        for info in allocation_info:
            key = info['stratum']
            final_actual = final_counts.get(key, 0)
            sampling_rate = final_actual / info['total'] * 100
            print(f"{key:<25} {info['total']:<8} {info['target']:<8} {final_actual:<8} {sampling_rate:>6.1f}%")
        
        print("-" * 80)
        print(f"{'总计':<25} {total_items:<8} {target_samples:<8} {len(sampled_data):<8} {len(sampled_data)/total_items*100:>6.1f}%")
        print("=" * 80)
        
        print(f"\n分层抽样完成！")
        print(f"从 {len(full_dataset)} 个问题中抽取了 {len(sampled_data)} 个代表性问题")
        
        return sampled_data
    
    def extract_answer_from_response(self, response_text: str) -> str:
        """从模型输出中提取最终答案"""
        if not response_text or not response_text.strip():
            return ""
        
        text = response_text.strip()
        
        # 匹配 \boxed{} 格式
        boxed_match = re.search(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text, re.DOTALL)
        if boxed_match:
            return boxed_match.group(1).strip()
        
        # 匹配最终答案行
        final_line = text.split('\n')[-1]
        if re.match(r'^[A-D]$', final_line):  # 选择题
            return final_line.strip()
        if re.match(r'^[\d\.]+$', final_line):  # 数值题
            return final_line.strip()
        
        return ""
    
    def normalize_math_expression(self, expr):
        """标准化数学表达式"""
        try:
            if not expr or not expr.strip():
                return ""
            
            # 基础清理
            expr = expr.strip()
            
            # 处理角度单位 - 统一转换
            expr = re.sub(r'(\d+)\s*\\?degrees?', r'\1°', expr, flags=re.IGNORECASE)
            expr = re.sub(r'(\d+)\s*\\?circ', r'\1°', expr)
            expr = re.sub(r'(\d+)\s*°', r'\1°', expr)  # 确保度数符号一致
            expr = re.sub(r'(\d+)\s*\^\\circ', r'\1°', expr)  # 处理^\\circ格式
            
            # 处理百分比
            expr = re.sub(r'(\d+(?:\.\d+)?)\s*\\?%', r'\1%', expr)
            
            # 处理LaTeX特殊格式
            expr = expr.replace(r'\left', '').replace(r'\right', '')
            expr = expr.replace(r'\dfrac', r'\frac').replace(r'\displaystyle', '')
            expr = expr.replace('\\pi', 'π').replace('\\Pi', 'π')
            
            # 处理LaTeX文本命令 - 移除\text{}包裹，保留内容
            expr = re.sub(r'\\text\{([^}]*)\}', r'\1', expr)
            
            # 移除boxed包装
            expr = re.sub(r'\\?boxed\{?(.*?)\}?', r'\1', expr)
            
            # 处理分数格式统一 - 先处理有大括号的，再处理无大括号的
            # 1. 处理 \frac{a}{b} 格式
            expr = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', expr)
            # 2. 处理 \frac{a}b 格式 (分子有括号，分母无括号)
            expr = re.sub(r'\\frac\{([^}]+)\}([^{}\s]+)', r'(\1)/(\2)', expr)
            # 3. 处理 \fraca{b} 格式 (分子无括号，分母有括号)
            expr = re.sub(r'\\frac([^{}\s]+)\{([^}]+)\}', r'(\1)/(\2)', expr)
            # 4. 处理 \fracab 格式 (都无括号) - 需要小心处理数字和字母
            expr = re.sub(r'\\frac([0-9]+)([0-9]+)', r'(\1)/(\2)', expr)
            
            # 移除多余空格
            expr = re.sub(r'\s+', '', expr)
            
            # 如果包含度数符号，直接返回（不需要sympy解析）
            if '°' in expr or 'degrees' in expr.lower() or '%' in expr:
                return expr
            
            # 尝试解析为SymPy表达式
            # 创建符号变量映射
            symbols = {'pi': sympy.pi, 'π': sympy.pi, 'e': sympy.E}
            parsed = parse_expr(expr, local_dict=symbols, transformations=transformations)
            return str(parsed)
        except:
            # 解析失败时返回简化后的原始字符串
            expr = re.sub(r'\\text\{([^}]*)\}', r'\1', expr)
            expr = re.sub(r'[^\w\d\.\-\+\*\/\^\(\)\,\\\[\]\{\}π°%]', '', expr)
            return expr.strip()

    def normalize_answer(self, ans):
        """标准化答案格式"""
        if not ans or not ans.strip():
            return ""
        
        # 基础清理
        ans = ans.strip()
        
        # 特殊处理角度答案
        if re.search(r'(\d+)\s*\^?\\?circ', ans):
            # 提取数值部分
            match = re.search(r'(\d+)', ans)
            if match:
                return f"{match.group(1)}°"
        
        # 特殊处理百分比
        if '%' in ans or 'percent' in ans.lower():
            match = re.search(r'(\d+(?:\.\d+)?)', ans)
            if match:
                return f"{match.group(1)}%"
        
        # 不要移除坐标点的括号！
        # 如果包含逗号且被括号包围，很可能是坐标，保留括号
        if not (',' in ans and ans.startswith('(') and ans.endswith(')')):
            # 只有不是坐标格式时，才考虑移除外层括号
            if ans.startswith('(') and ans.endswith(')') and ans.count('(') == 1 and ans.count(')') == 1:
                ans = ans[1:-1]
        
        # 特殊处理坐标格式
        if re.match(r'^\d+,\s*\\?frac\{.*?\}', ans):
            parts = ans.split(',', 1)
            return f"({parts[0].strip()}, {parts[1].strip()})"
        
        return ans

    def check_answer_correctness(self, pred, true):
        """比较两个答案是否等价"""
        if not pred or not true:
            return False
        
        # 1. 直接比较
        if pred == true:
            return True
        
        # 2. 标准化后比较
        norm_pred = self.normalize_math_expression(self.normalize_answer(pred))
        norm_true = self.normalize_math_expression(self.normalize_answer(true))
        
        if norm_pred == norm_true:
            return True
        
        # 3. 特殊情况处理
        
        # 区间表达式特殊处理
        # 检查是否包含区间表达式的特征
        if ('\\cup' in norm_pred or '\\cup' in norm_true or 
            'cup' in norm_pred or 'cup' in norm_true or
            ('infty' in norm_pred or 'infty' in norm_true) or
            ('∞' in norm_pred or '∞' in norm_true) or
            ('interval' in norm_pred.lower() or 'interval' in norm_true.lower())):
            # 对于区间表达式，只有完全匹配才认为正确
            # 移除空格进行严格比较
            pred_clean = re.sub(r'\s+', '', norm_pred)
            true_clean = re.sub(r'\s+', '', norm_true)
            return pred_clean == true_clean
        
        # 角度单位比较
        if ('°' in norm_pred or '°' in norm_true):
            # 提取数值部分进行比较
            pred_num = re.search(r'(\d+(?:\.\d+)?)', norm_pred)
            true_num = re.search(r'(\d+(?:\.\d+)?)', norm_true)
            if pred_num and true_num:
                try:
                    return float(pred_num.group(1)) == float(true_num.group(1))
                except:
                    pass
        
        # 百分比比较
        if ('%' in norm_pred or '%' in norm_true):
            pred_num = re.search(r'(\d+(?:\.\d+)?)', norm_pred)
            true_num = re.search(r'(\d+(?:\.\d+)?)', norm_true)
            if pred_num and true_num:
                try:
                    return float(pred_num.group(1)) == float(true_num.group(1))
                except:
                    pass
        
        # 4. 数值计算比较
        try:
            # 创建符号变量
            symbols = {'pi': sympy.pi, 'π': sympy.pi, 'e': sympy.E}
            
            # 移除单位后进行数值比较
            pred_clean = re.sub(r'[°%]', '', norm_pred)
            true_clean = re.sub(r'[°%]', '', norm_true)
            
            # 解析表达式
            expr_pred = parse_expr(pred_clean, local_dict=symbols, transformations=transformations)
            expr_true = parse_expr(true_clean, local_dict=symbols, transformations=transformations)
            
            # 数值计算比较
            diff = sympy.simplify(expr_pred - expr_true)
            return diff == 0 or abs(float(diff.evalf())) < 1e-10
        except Exception as e:
            # 最后尝试纯数值比较
            try:
                # 提取所有数字
                pred_nums = re.findall(r'(\d+(?:\.\d+)?)', norm_pred)
                true_nums = re.findall(r'(\d+(?:\.\d+)?)', norm_true)
                
                if pred_nums and true_nums:
                    # 比较主要数值
                    return float(pred_nums[0]) == float(true_nums[0])
            except:
                pass
            
            return False
    
    def run_single_problem_experiment(self, problem_data: Dict, 
                                     intervention_configs: List[Dict]) -> Dict:
        """对单个问题运行实验"""
        
        problem = problem_data['problem']
        correct_answer = problem_data['answer']
        
        # 使用ChatML格式构建提示，适配Qwen模型
        base_prompt = (
            f"<|im_start|>system\n你是一个数学专家，请解决以下数学问题。"
            f"答案必须使用 \\boxed{{}} 包裹。<|im_end|>\n"
            f"<|im_start|>user\n{problem}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        results = {
            'problem': problem,
            'correct_answer': correct_answer,
            # 兼容不同数据集格式
            'subject': problem_data.get('subject', problem_data.get('problem_type', 'Unknown')),
            'level': problem_data.get('level', 'Unknown'),
            'problem_type': problem_data.get('problem_type', 'Unknown'),
            'question_type': problem_data.get('question_type', 'Unknown'),
            'source': problem_data.get('source', 'Unknown'),
            'experiments': {}
        }
        
        for config in intervention_configs:
            exp_name = self.get_experiment_name(config)
            
            try:
                # 设置干预
                self.controller.set_dimension_intervention(
                    layer_idx=config.get('target_layer', config.get('layer', 14)),
                    dimensions=config.get('target_dimensions', config.get('dimensions', [16, 18])),
                    intervention_type=config.get('intervention_type', 'gaussian_replace'),
                    scale_factor=config.get('scale_factor', 1.0),
                    gaussian_mean=config.get('gaussian_mean', 0.0),
                    gaussian_std=config.get('gaussian_std', 1.0),
                    intervene_on_prompt=False
                )
                
                # 运行双模式生成
                experiment_results = self.controller.generate_with_dual_mode_intervention(
                    prompt=base_prompt,
                    max_new_tokens=config.get('max_new_tokens', 512),
                    target_layer=config.get('target_layer', config.get('layer', 14)),
                    target_dimensions=config.get('target_dimensions', config.get('dimensions', [16, 18])),
                    nothink_temperature=config.get('temperature', 0.7),
                    nothink_top_p=config.get('top_p', 0.8),
                    think_temperature=config.get('temperature', 0.6),
                    think_top_p=config.get('top_p', 0.95),
                    think_do_sample=True,
                    nothink_do_sample=True
                )
                
                exp_result = {}
                
                # 处理实验结果
                if experiment_results:
                    # 处理 nothink 模式
                    if 'nothink_mode' in experiment_results:
                        nothink_data = experiment_results['nothink_mode']
                        original_text = nothink_data.get('original_solution', '')
                        intervention_text = nothink_data.get('intervention_solution', '')
                        
                        # 提取答案
                        original_answer = self.extract_answer_from_response(original_text)
                        intervention_answer = self.extract_answer_from_response(intervention_text)
                        
                        # 检查正确性
                        original_correct = self.check_answer_correctness(original_answer, correct_answer)
                        intervention_correct = self.check_answer_correctness(intervention_answer, correct_answer)
                        
                        exp_result['nothink_mode'] = {
                            'original_response': original_text,
                            'intervention_response': intervention_text,
                            'original_answer': original_answer,
                            'intervention_answer': intervention_answer,
                            'original_correct': original_correct,
                            'intervention_correct': intervention_correct
                        }
                    
                    # 处理 think 模式
                    if 'think_mode' in experiment_results:
                        think_data = experiment_results['think_mode']
                        original_text = think_data.get('original_solution', '')
                        intervention_text = think_data.get('intervention_solution', '')
                        
                        # 提取答案
                        original_answer = self.extract_answer_from_response(original_text)
                        intervention_answer = self.extract_answer_from_response(intervention_text)
                        
                        # 检查正确性
                        original_correct = self.check_answer_correctness(original_answer, correct_answer)
                        intervention_correct = self.check_answer_correctness(intervention_answer, correct_answer)
                        
                        exp_result['think_mode'] = {
                            'original_response': original_text,
                            'intervention_response': intervention_text,
                            'original_answer': original_answer,
                            'intervention_answer': intervention_answer,
                            'original_correct': original_correct,
                            'intervention_correct': intervention_correct
                        }
                
                results['experiments'][exp_name] = exp_result
                
                # 清除钩子，为下一个实验准备
                self.controller.clear_hooks()
                
            except Exception as e:
                print(f"实验 {exp_name} 运行失败: {e}")
                import traceback
                traceback.print_exc()
                results['experiments'][exp_name] = {'error': str(e)}
                # 即使出错也要清除钩子
                self.controller.clear_hooks()
        
        return results
    
    def get_experiment_name(self, config: Dict) -> str:
        """生成实验名称"""
        if not config.get('intervention', True):
            return "no_intervention"
        
        intervention_type = config.get('intervention_type', 'gaussian_replace')
        if intervention_type == 'gaussian_replace':
            return f"intervention_{intervention_type}_mean{config.get('gaussian_mean', 0)}_std{config.get('gaussian_std', 1)}"
        elif intervention_type == 'scale':
            return f"intervention_{intervention_type}_factor{config.get('scale_factor', 1.0)}"
        else:
            return f"intervention_{intervention_type}"
    
    def calculate_config_stats(self, results: List[Dict], config_name: str) -> Dict:
        """计算特定配置的统计信息"""
        stats = {
            'nothink_orig': 0, 'nothink_interv': 0,
            'think_orig': 0, 'think_interv': 0
        }
        
        nothink_count = think_count = 0
        
        for result in results:
            if config_name in result['experiments']:
                exp_result = result['experiments'][config_name]
                
                if 'nothink_mode' in exp_result:
                    nothink_count += 1
                    if exp_result['nothink_mode']['original_correct']:
                        stats['nothink_orig'] += 1
                    if exp_result['nothink_mode']['intervention_correct']:
                        stats['nothink_interv'] += 1
                
                if 'think_mode' in exp_result:
                    think_count += 1
                    if exp_result['think_mode']['original_correct']:
                        stats['think_orig'] += 1
                    if exp_result['think_mode']['intervention_correct']:
                        stats['think_interv'] += 1
        
        # 转换为百分比
        if nothink_count > 0:
            stats['nothink_orig'] = stats['nothink_orig'] / nothink_count * 100
            stats['nothink_interv'] = stats['nothink_interv'] / nothink_count * 100
        
        if think_count > 0:
            stats['think_orig'] = stats['think_orig'] / think_count * 100
            stats['think_interv'] = stats['think_interv'] / think_count * 100
        
        return stats

    @staticmethod
    def run_distributed_experiment(model_path: str, dataset_path: str, 
                                  intervention_configs: List[Dict],
                                  max_samples: int = None,
                                  output_dir: str = "math_intervention_results",
                                  num_gpus: int = None,
                                  gpu_ids: List[int] = None,
                                  use_stratified_sampling: bool = False,
                                  target_samples: int = 100,
                                  random_seed: int = 42,
                                  openr1_split: str = "default",
                                  filter_geometry: bool = True,
                                  single_file_index: int = None) -> List[Dict]:
        """运行分布式实验"""
        
        # 检测可用GPU
        try:
            available_gpus = list(range(torch.cuda.device_count()))
            print(f"检测到 {len(available_gpus)} 个GPU: {available_gpus}")
        except:
            available_gpus = []
            print("未检测到GPU，将使用CPU模式")
        
        if not available_gpus:
            print("警告：没有可用的GPU，无法运行分布式实验")
            return []
        
        # 确定要使用的GPU列表
        if gpu_ids:
            # 验证指定的GPU是否可用
            gpu_list = [gpu for gpu in gpu_ids if gpu in available_gpus]
            if not gpu_list:
                print(f"指定的GPU {gpu_ids} 都不可用，使用所有可用GPU")
                gpu_list = available_gpus
            else:
                print(f"使用指定的GPU: {gpu_list}")
        elif num_gpus:
            gpu_list = available_gpus[:min(num_gpus, len(available_gpus))]
            print(f"使用前 {len(gpu_list)} 个GPU: {gpu_list}")
        else:
            gpu_list = available_gpus
            print(f"使用所有可用GPU: {gpu_list}")
        
        # 创建临时实验实例来加载数据集
        temp_experiment = MathInterventionExperiment(model_path)
        
        # 根据参数选择加载方式
        if use_stratified_sampling:
            print(f"使用分层抽样模式，目标样本数: {target_samples}")
            
            # 检查是否是OpenR1-Math数据集
            if 'OpenR1-Math' in dataset_path:
                dataset = temp_experiment.stratified_sample_openr1_dataset(
                    dataset_path, openr1_split, filter_geometry, target_samples, random_seed,
                    single_file_index
                )
            else:
                dataset = temp_experiment.stratified_sample_dataset(
                    dataset_path, target_samples, random_seed, single_file_index
                )
        else:
            # 检查是否是OpenR1-Math数据集
            if 'OpenR1-Math' in dataset_path:
                dataset = temp_experiment.load_openr1_math_dataset(
                    dataset_path, openr1_split, filter_geometry, max_samples, single_file_index
                )
            else:
                dataset = temp_experiment.load_math_dataset(dataset_path, max_samples, single_file_index)
        
        del temp_experiment  # 释放内存
        
        # 将数据集分块
        chunk_size = math.ceil(len(dataset) / len(gpu_list))
        chunks = [dataset[i:i + chunk_size] for i in range(0, len(dataset), chunk_size)]
        
        print(f"数据集分为 {len(chunks)} 块，每块约 {chunk_size} 个问题")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 准备共享参数
        shared_args = {
            'max_samples': max_samples,
            'dataset_path': dataset_path,
            'use_stratified_sampling': use_stratified_sampling,
            'target_samples': target_samples,
            'random_seed': random_seed,
            'openr1_split': openr1_split,
            'filter_geometry': filter_geometry,
            'single_file_index': single_file_index
        }
        
        # 使用ProcessPoolExecutor进行分布式处理
        all_results = []
        
        with ProcessPoolExecutor(max_workers=len(gpu_list)) as executor:
            # 提交所有任务
            futures = []
            for i, chunk in enumerate(chunks):
                gpu_id = gpu_list[i % len(gpu_list)]  # 使用指定的GPU列表循环分配
                future = executor.submit(
                    MathInterventionExperiment._run_gpu_chunk,
                    gpu_id, model_path, chunk, intervention_configs, 
                    i, output_dir, shared_args
                )
                futures.append(future)
            
            # 收集结果
            for future in as_completed(futures):
                try:
                    chunk_results = future.result()
                    all_results.extend(chunk_results)
                except Exception as e:
                    print(f"处理GPU块时出错: {e}")
                    import traceback
                    traceback.print_exc()
        
        # 按问题索引排序结果
        all_results.sort(key=lambda x: x.get('problem_index', 0))
        
        # 保存最终结果
        final_file = os.path.join(output_dir, "final_results.json")
        with open(final_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        # 生成统计报告
        MathInterventionExperiment._generate_statistics_report(all_results, output_dir)
        
        print(f"分布式实验完成，处理了 {len(all_results)} 个问题")
        print(f"结果保存在: {output_dir}")
        
        return all_results
    
    @staticmethod
    def _run_gpu_chunk(gpu_id: int, model_path: str, dataset_chunk: List, 
                      intervention_configs: List[Dict], chunk_id: int, 
                      output_dir: str, shared_args: Dict) -> List[Dict]:
        """在单个GPU上处理数据块的静态方法"""
        try:
            # 设置GPU设备
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            
            # 初始化实验（在子进程中重新初始化）
            experiment = MathInterventionExperiment(model_path)
            
            results = []
            
            print(f"GPU {gpu_id} 开始处理 {len(dataset_chunk)} 个问题")
            
            for i, problem_data in enumerate(dataset_chunk):
                try:
                    result = experiment.run_single_problem_experiment(problem_data, intervention_configs)
                    result['problem_index'] = chunk_id * len(dataset_chunk) + i
                    results.append(result)
                    
                    # 每5个问题保存一次临时结果
                    if (i + 1) % 5 == 0:
                        temp_file = os.path.join(output_dir, f"chunk_{chunk_id}_temp_{i+1}.json")
                        with open(temp_file, 'w', encoding='utf-8') as f:
                            json.dump(results, f, ensure_ascii=False, indent=2)
                        print(f"GPU {gpu_id} 已完成 {i+1}/{len(dataset_chunk)} 个问题")
                    
                except Exception as e:
                    print(f"GPU {gpu_id} 处理问题 {i} 时出错: {e}")
                    continue
            
            # 保存该GPU的最终结果
            chunk_file = os.path.join(output_dir, f"chunk_{chunk_id}_results.json")
            with open(chunk_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"GPU {gpu_id} 完成处理，结果保存到 {chunk_file}")
            return results
            
        except Exception as e:
            print(f"GPU {gpu_id} 处理过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    @staticmethod
    def _generate_statistics_report(results: List[Dict], output_dir: str):
        """生成统计报告"""
        report_file = os.path.join(output_dir, "statistics_report.md")
        
        # 收集所有实验配置
        all_configs = set()
        for result in results:
            for exp_name in result['experiments'].keys():
                all_configs.add(exp_name)
        
        all_configs = sorted(list(all_configs))
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# MATH500数据集神经干预实验统计报告\n\n")
            f.write(f"## 实验概览\n\n")
            f.write(f"- **总问题数**: {len(results)}\n")
            f.write(f"- **实验配置数**: {len(all_configs)}\n\n")
            
            # 总体准确率统计
            f.write("## 总体准确率统计\n\n")
            f.write("| 实验配置 | 模式 | NoThink原始正确率 | NoThink干预正确率 | Think原始正确率 | Think干预正确率 |\n")
            f.write("|----------|------|------------------|------------------|----------------|----------------|\n")
            
            for config_name in all_configs:
                # 统计该配置下的准确率
                nothink_orig_correct = 0
                nothink_interv_correct = 0
                think_orig_correct = 0
                think_interv_correct = 0
                total_nothink = 0
                total_think = 0
                
                for result in results:
                    if config_name in result['experiments']:
                        exp_result = result['experiments'][config_name]
                        
                        if 'nothink_mode' in exp_result:
                            total_nothink += 1
                            if exp_result['nothink_mode']['original_correct']:
                                nothink_orig_correct += 1
                            if exp_result['nothink_mode']['intervention_correct']:
                                nothink_interv_correct += 1
                        
                        if 'think_mode' in exp_result:
                            total_think += 1
                            if exp_result['think_mode']['original_correct']:
                                think_orig_correct += 1
                            if exp_result['think_mode']['intervention_correct']:
                                think_interv_correct += 1
                
                nothink_orig_acc = (nothink_orig_correct / total_nothink * 100) if total_nothink > 0 else 0
                nothink_interv_acc = (nothink_interv_correct / total_nothink * 100) if total_nothink > 0 else 0
                think_orig_acc = (think_orig_correct / total_think * 100) if total_think > 0 else 0
                think_interv_acc = (think_interv_correct / total_think * 100) if total_think > 0 else 0
                
                f.write(f"| {config_name} | 双模式 | {nothink_orig_acc:.1f}% ({nothink_orig_correct}/{total_nothink}) | {nothink_interv_acc:.1f}% ({nothink_interv_correct}/{total_nothink}) | {think_orig_acc:.1f}% ({think_orig_correct}/{total_think}) | {think_interv_acc:.1f}% ({think_interv_correct}/{total_think}) |\n")
            
            f.write("\n")
        
        print(f"统计报告已保存到: {report_file}")


def main():
    # 设置multiprocessing启动方法为spawn以支持CUDA（必须在任何CUDA操作之前）
    if mp.get_start_method(allow_none=True) != 'spawn':
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # 如果已经设置过了就忽略
    
    parser = argparse.ArgumentParser(description='MATH数据集神经干预实验')
    parser.add_argument('--model_path', type=str, 
                        default='/data4/huguangyi/models/Qwen/Qwen3-0.6B',
                        help='模型路径')
    parser.add_argument('--dataset_path', type=str,
                        default='/data4/huguangyi/datasets/OpenR1-Math',
                        help='数据集路径（支持MATH或OpenR1-Math格式）')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='最大样本数（用于测试）')
    parser.add_argument('--target_layer', type=int, default=14,
                        help='目标层')
    parser.add_argument('--target_dimensions', type=str, default='1, 2, 3, 5, 8, 9, 13, 16, 18, 28, 46, 61, 77, 81, 86, 92, 97, 103, 131, 139, 196, 230, 242, 306, 310, 402, 566, 569, 604, 654, 656, 663, 666, 671, 686, 700, 703, 810, 816, 826, 832, 840, 896',
                        help='目标维度，逗号分隔')
    parser.add_argument('--output_dir', type=str, default='math_intervention_results_think_1epoch_43nodes_geometry',
                        help='输出目录')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')
    parser.add_argument('--gpu_ids', type=str, default='0,3,5,6,7',
                        help='使用的GPU ID列表，逗号分隔，例如"0,2,4"。如果不指定，使用所有可用GPU')
    parser.add_argument('--max_new_tokens', type=int, default=32768,
                        help='最大生成token数')
    parser.add_argument('--num_gpus', type=int, default=None,
                        help='使用的GPU数量（从GPU 0开始）')
    
    # 分层抽样相关参数
    parser.add_argument('--use_stratified_sampling', action='store_true',
                        help='使用分层抽样模式，按level和subject进行分层抽样')
    parser.add_argument('--target_samples', type=int, default=100,
                        help='分层抽样的目标样本数量（默认100）')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='分层抽样的随机种子（默认42）')
    
    # OpenR1-Math数据集相关参数
    parser.add_argument('--openr1_split', type=str, default='default',
                        choices=['default', 'extended', 'all'],
                        help='OpenR1-Math数据集分割类型（default, extended, all）')
    parser.add_argument('--filter_geometry', action='store_true', default=True,
                        help='是否只筛选几何类型的问题（默认开启）')
    parser.add_argument('--no_filter_geometry', dest='filter_geometry', action='store_false',
                        help='关闭几何筛选，使用所有类型的数学问题')
    
    # 单文件模式参数
    parser.add_argument('--single_file_index', type=int, default=0,
                        help='只加载指定索引的单个parquet文件（0-9，None表示加载所有文件）。例如：0表示只加载train-00000-of-00010.parquet')
    
    args = parser.parse_args()
    
    # 解析目标维度
    target_dimensions = [int(d.strip()) for d in args.target_dimensions.split(',')]
    
    # 解析GPU ID列表
    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpu_ids.split(',')]
        print(f"指定GPU列表: {gpu_ids}")
    
    # 定义实验配置
    intervention_configs = [
        # 高斯替换干预
        {
            'intervention': True,
            'target_layer': args.target_layer,
            'target_dimensions': target_dimensions,
            'intervention_type': 'gaussian_replace',
            'gaussian_mean': 0,
            'gaussian_std': 0,
            'max_new_tokens': args.max_new_tokens
        }
    ]
    
    # 判断数据集类型并打印相应信息
    if 'OpenR1-Math' in args.dataset_path:
        print(f"开始OpenR1-Math神经干预实验（分布式模式）")
        print(f"数据集分割: {args.openr1_split}")
        print(f"几何筛选: {'开启' if args.filter_geometry else '关闭'}")
        if args.single_file_index is not None:
            print(f"单文件模式: 只加载文件索引 {args.single_file_index} (train-{args.single_file_index:05d}-of-00010.parquet)")
    else:
        print(f"开始MATH数据集神经干预实验（分布式模式）")
    
    print(f"模型: {args.model_path}")
    print(f"数据集: {args.dataset_path}")
    
    if args.use_stratified_sampling:
        print(f"使用分层抽样模式:")
        print(f"  - 目标样本数: {args.target_samples}")
        print(f"  - 随机种子: {args.random_seed}")
    else:
        print(f"最大样本数: {args.max_samples}")
    
    print(f"目标层: {args.target_layer}")
    print(f"目标维度: {target_dimensions}")
    print(f"实验配置数: {len(intervention_configs)}")
    
    # 运行分布式实验
    results = MathInterventionExperiment.run_distributed_experiment(
        args.model_path,
        args.dataset_path,
        intervention_configs,
        args.max_samples,
        args.output_dir,
        args.num_gpus,
        gpu_ids,
        args.use_stratified_sampling,
        args.target_samples,
        args.random_seed,
        args.openr1_split,
        args.filter_geometry,
        args.single_file_index
    )
    
    if args.use_stratified_sampling:
        print(f"\n分层抽样实验完成！共处理 {len(results)} 个问题。")
    else:
        print(f"\n实验完成！共处理 {len(results)} 个问题。")
    print(f"结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main() 