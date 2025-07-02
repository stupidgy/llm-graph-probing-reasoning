#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
import torch
import argparse
from typing import List, Dict, Tuple
from tqdm import tqdm
import copy
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
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
    
    def load_math_dataset(self, dataset_path: str, max_samples: int = None) -> List[Dict]:
        """加载MATH数据集"""
        print(f"正在加载数据集: {dataset_path}")
        
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
            # 处理LaTeX特殊格式
            expr = expr.replace(r'\left', '').replace(r'\right', '')
            expr = expr.replace(r'\dfrac', r'\frac').replace(r'\displaystyle', '')
            expr = expr.replace('\\pi', 'π')  # 统一使用Unicode π
            
            # 处理LaTeX文本命令 - 移除\text{}包裹，保留内容
            expr = re.sub(r'\\text\{([^}]*)\}', r'\1', expr)
            
            # 移除空格和多余括号
            expr = re.sub(r'\s+', '', expr)
            expr = re.sub(r'\\?boxed\{?(.*?)\}?', r'\1', expr)  # 移除boxed
            
            # 尝试解析为SymPy表达式
            parsed = parse_expr(expr, transformations=transformations)
            return str(parsed)
        except:
            # 解析失败时返回简化后的原始字符串
            # 仍然要移除\text{}
            expr = re.sub(r'\\text\{([^}]*)\}', r'\1', expr)
            expr = re.sub(r'[^\w\d\.\-\+\*\/\^\(\)\,\\\[\]\{\}π]', '', expr)
            return expr.strip()

    def normalize_answer(self, ans):
        """标准化答案格式"""
        # 基础清理
        ans = ans.strip()
        
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
        # 1. 直接比较
        if pred == true:
            return True
        
        # 2. 标准化后比较
        norm_pred = self.normalize_math_expression(self.normalize_answer(pred))
        norm_true = self.normalize_math_expression(self.normalize_answer(true))
        
        if norm_pred == norm_true:
            return True
        
        # 3. 数值计算比较
        try:
            # 创建符号变量
            symbols = {'pi': sympy.pi, 'π': sympy.pi}
            
            # 解析表达式
            expr_pred = parse_expr(norm_pred, local_dict=symbols, transformations=transformations)
            expr_true = parse_expr(norm_true, local_dict=symbols, transformations=transformations)
            
            # 数值计算比较
            return sympy.simplify(expr_pred - expr_true) == 0
        except:
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
            'subject': problem_data.get('subject', 'Unknown'),
            'level': problem_data.get('level', 'Unknown'),
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
                    think_top_p=config.get('top_p', 0.95)
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
                                  gpu_ids: List[int] = None) -> List[Dict]:
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
        dataset = temp_experiment.load_math_dataset(dataset_path, max_samples)
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
            'dataset_path': dataset_path
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
    
    parser = argparse.ArgumentParser(description='MATH500数据集神经干预实验')
    parser.add_argument('--model_path', type=str, 
                        default='/data4/huguangyi/models/Qwen/Qwen3-0.6B',
                        help='模型路径')
    parser.add_argument('--dataset_path', type=str,
                        default='/data4/huguangyi/datasets/MATH500/test.jsonl',
                        help='MATH数据集路径')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='最大样本数（用于测试）')
    parser.add_argument('--target_layer', type=int, default=14,
                        help='目标层')
    parser.add_argument('--target_dimensions', type=str, default='16,18',
                        help='目标维度，逗号分隔')
    parser.add_argument('--output_dir', type=str, default='math_intervention_results',
                        help='输出目录')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')
    parser.add_argument('--gpu_ids', type=str, default='2,7',
                        help='使用的GPU ID列表，逗号分隔，例如"0,2,4"。如果不指定，使用所有可用GPU')
    parser.add_argument('--max_new_tokens', type=int, default=32768,
                        help='最大生成token数')
    parser.add_argument('--num_gpus', type=int, default=None,
                        help='使用的GPU数量（从GPU 0开始）')
    
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
    
    print(f"开始MATH500神经干预实验（分布式模式）")
    print(f"模型: {args.model_path}")
    print(f"数据集: {args.dataset_path}")
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
        gpu_ids
    )
    
    print(f"\n实验完成！共处理 {len(results)} 个问题。")
    print(f"结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main() 