import os
import json
import torch
import pandas as pd
import glob
from tqdm import tqdm
import argparse
import time
import random
import multiprocessing as mp
from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="使用Qwen3生成数学问题解答")
    parser.add_argument("--model_path", type=str, default="/data4/huguangyi/models/Qwen/Qwen3-0.6B", help="模型路径")
    parser.add_argument("--dataset_path", type=str, default="/data4/huguangyi/datasets/AI-MO/NuminaMath-CoT", help="数据集路径")
    parser.add_argument("--output_path", type=str, default="./outputs_nothink/qwen3_math_solutions.jsonl", help="输出文件路径")
    parser.add_argument("--batch_size", type=int, default=32, help="批处理大小")
    parser.add_argument("--max_samples", type=int, default=10000, help="最大处理样本数量，-1为全部")
    parser.add_argument("--max_new_tokens", type=int, default=32768, help="最大生成token数")
    parser.add_argument("--retry_count", type=int, default=3, help="出错时重试次数")
    parser.add_argument("--use_multi_gpu", action="store_true", help="使用多GPU并行处理数据集")
    parser.add_argument("--gpu_ids", type=str, default="6,7", help="指定用于数据并行的GPU ID，用逗号分隔")
    return parser.parse_args()

def load_numina_math_dataset(dataset_path, max_samples=-1):
    """加载NuminaMath-CoT数据集"""
    data_dir = os.path.join(dataset_path, "data")
    if not os.path.exists(data_dir):
        raise ValueError(f"找不到数据目录: {data_dir}")
    
    # 加载第一个分片
    all_files = sorted(glob.glob(os.path.join(data_dir, "train-*-of-*.parquet")))
    if not all_files:
        raise ValueError(f"找不到数据文件在 {data_dir}")
    
    print(f"加载数据文件: {all_files[0]}")
    df = pd.read_parquet(all_files[0])
    
    # 提取问题
    problems = df["problem"].tolist()
    
    # 获取前几个
    if max_samples > 0:
        problems = problems[:max_samples]
        
    return problems

def get_think_token_id(tokenizer):
    """获取</think>的token ID"""
    # 针对Qwen3模型获取</think>标记的ID
    try:
        think_token = "</think>"
        think_token_id = tokenizer.convert_tokens_to_ids(think_token)
        if think_token_id == tokenizer.unk_token_id:
            # 如果转换结果是unk token，尝试其他方式
            think_token_id = None
            
            # 方法1: 尝试编码整个标记并获取最后一个token
            tokens = tokenizer.tokenize(think_token)
            if tokens and tokens[-1] != tokenizer.unk_token:
                think_token_id = tokenizer.convert_tokens_to_ids(tokens[-1])
            
            # 方法2: 检查是否有特殊标记列表
            if hasattr(tokenizer, 'special_tokens_map') and 'thinking' in tokenizer.special_tokens_map:
                think_token = tokenizer.special_tokens_map['thinking']
                think_token_id = tokenizer.convert_tokens_to_ids(think_token)
            
            # 如果都找不到，使用默认值
            if think_token_id is None or think_token_id == tokenizer.unk_token_id:
                # Qwen3的</think>标记ID可能是151668
                print("无法确定</think>标记ID，使用默认值151668")
                think_token_id = 151668
        
        print(f"</think>标记的ID: {think_token_id}")
        return think_token_id
    except Exception as e:
        print(f"获取</think>标记ID失败: {e}")
        return 151668  # 默认值

def save_results(result, output_path, rank=0):
    """保存结果到jsonl文件"""
    # 创建带有rank的输出路径
    if rank > 0:
        base_dir = os.path.dirname(output_path)
        base_name = os.path.basename(output_path)
        output_path = os.path.join(base_dir, f"rank{rank}_{base_name}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    return output_path

def process_subset(problems, args, rank, total_ranks):
    """处理问题子集的函数，将在各个进程中运行"""
    # 设置当前进程使用的GPU
    if args.gpu_ids:
        gpu_ids = args.gpu_ids.split(',')
        if rank < len(gpu_ids):
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids[rank]
            print(f"进程 {rank}: 使用GPU {gpu_ids[rank]}")
        else:
            print(f"警告: 进程 {rank} 没有对应的GPU，将使用CPU")
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # 加载模型和分词器
    print(f"进程 {rank}: 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    
    # 获取</think>标记的ID
    think_token_id = get_think_token_id(tokenizer)
    
    # 进程输出路径
    process_output_path = args.output_path
    if rank > 0:
        dir_name = os.path.dirname(args.output_path)
        base_name = os.path.basename(args.output_path)
        process_output_path = os.path.join(dir_name, f"rank{rank}_{base_name}")
    
    # 检查已处理数量
    processed_count = 0
    if os.path.exists(process_output_path):
        try:
            with open(process_output_path, "r", encoding="utf-8") as f:
                processed_count = len(f.readlines())
            print(f"进程 {rank}: 发现已处理 {processed_count} 个问题，将从第 {processed_count} 个继续")
        except:
            processed_count = 0
    
    # 分批处理
    results = []
    start_time = time.time()
    for batch_start in range(0, len(problems), args.batch_size):
        # 跳过已处理的
        if batch_start < processed_count:
            continue
            
        batch_end = min(batch_start + args.batch_size, len(problems))
        batch_problems = problems[batch_start:batch_end]
        
        batch_messages = [[{"role": "user", "content": problem}] for problem in batch_problems]
        batch_texts = [
            tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=False
            ) for messages in batch_messages
        ]
        
        # 处理每个样本
        for i, (problem, text) in enumerate(zip(batch_problems, batch_texts)):
            global_idx = batch_start + i
            print(f"进程 {rank}: 处理问题 {global_idx + 1}/{len(problems)}")
            
            retry_count = 0
            success = False
            
            while retry_count < args.retry_count and not success:
                try:
                    # 准备输入
                    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
                    
                    # 生成回答
                    with torch.no_grad():
                        generated_ids = model.generate(
                            **model_inputs,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.8,
                            top_k=20
                        )
                    
                    # 提取生成的内容
                    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
                    
                    # 尝试分离思考内容和最终答案
                    try:
                        # </think>标记的索引位置
                        index = len(output_ids) - output_ids[::-1].index(think_token_id)
                    except ValueError:
                        print(f"进程 {rank}: 未找到</think>标记，将完整输出作为解答")
                        index = 0
                    
                    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
                    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
                    
                    # 保存结果
                    result = {
                        "problem_idx": global_idx,
                        "problem": problem,
                        "thinking": thinking_content,
                        "solution": content,
                    }
                    results.append(result)
                    
                    # 实时保存每个结果
                    save_results(result, args.output_path, rank)
                    
                    success = True
                    
                except Exception as e:
                    retry_count += 1
                    print(f"进程 {rank}: 处理失败 ({retry_count}/{args.retry_count}): {e}")
                    time.sleep(1)  # 等待一秒后重试
            
            if not success:
                print(f"进程 {rank}: 跳过问题 {global_idx + 1} - 多次尝试后失败")
                # 记录失败的样本
                with open(f"{process_output_path}.failed", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"problem_idx": global_idx, "problem": problem, "error": "多次尝试后失败"}, ensure_ascii=False) + "\n")
            
            # 打印进度
            elapsed_time = time.time() - start_time
            avg_time = elapsed_time / (global_idx + 1) if global_idx >= batch_start else 0
            remaining = (len(problems) - global_idx - 1) * avg_time if avg_time > 0 else 0
            print(f"进程 {rank}: 进度: {global_idx + 1}/{len(problems)}, 已用时间: {elapsed_time:.1f}s, 平均每样本: {avg_time:.1f}s, 预计剩余时间: {remaining:.1f}s")
        
        # 每批次结束后清空GPU缓存
        torch.cuda.empty_cache()
    
    print(f"进程 {rank}: 处理完成，共 {len(results)} 个结果已保存到 {process_output_path}")
    return process_output_path

def split_problems(problems, num_processes):
    """将问题数据集分割为多个子集"""
    avg = len(problems) // num_processes
    remainder = len(problems) % num_processes
    
    result = []
    start = 0
    for i in range(num_processes):
        end = start + avg + (1 if i < remainder else 0)
        result.append(problems[start:end])
        start = end
    
    return result

def merge_results(output_path, num_processes):
    """合并多个进程的结果文件"""
    print("合并所有进程的结果...")
    all_results = []
    
    # 读取主输出文件
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                all_results.append(json.loads(line))
    
    # 读取其他进程的输出文件
    for rank in range(1, num_processes):
        dir_name = os.path.dirname(output_path)
        base_name = os.path.basename(output_path)
        proc_output = os.path.join(dir_name, f"rank{rank}_{base_name}")
        
        if os.path.exists(proc_output):
            with open(proc_output, "r", encoding="utf-8") as f:
                for line in f:
                    all_results.append(json.loads(line))
    
    # 按问题索引排序
    all_results.sort(key=lambda x: x.get("problem_idx", 0))
    
    # 保存合并后的结果
    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"合并完成，共 {len(all_results)} 个结果已保存到 {output_path}")
    
    # 可选：删除临时文件
    for rank in range(1, num_processes):
        dir_name = os.path.dirname(output_path)
        base_name = os.path.basename(output_path)
        proc_output = os.path.join(dir_name, f"rank{rank}_{base_name}")
        
        if os.path.exists(proc_output):
            try:
                os.remove(proc_output)
                print(f"已删除临时文件: {proc_output}")
            except:
                print(f"无法删除临时文件: {proc_output}")

def main():
    args = parse_args()
    
    # 设置随机种子以确保结果可复现
    random.seed(42)
    torch.manual_seed(42)
    
    # 加载数据集
    print("加载数据集...")
    try:
        problems = load_numina_math_dataset(args.dataset_path, args.max_samples)
        print(f"加载了 {len(problems)} 个问题")
    except Exception as e:
        print(f"加载数据集失败: {e}")
        return

    # 如果使用多GPU并行处理
    if args.use_multi_gpu:
        gpu_ids = args.gpu_ids.split(',')
        num_processes = len(gpu_ids)
        
        if num_processes <= 1:
            print("指定的GPU不足两个，将使用单进程处理")
            args.use_multi_gpu = False
        else:
            print(f"使用 {num_processes} 个进程进行数据并行处理")
            
            # 将数据集分割为多个子集
            problem_subsets = split_problems(problems, num_processes)
            
            # 创建多个进程处理各自的子集
            with mp.Pool(num_processes) as pool:
                pool.starmap(
                    process_subset,
                    [(subset, args, i, num_processes) for i, subset in enumerate(problem_subsets)]
                )
            
            # 合并结果
            merge_results(args.output_path, num_processes)
            return
    
    # 单进程处理（默认）
    # 获取已处理的样本数
    processed_count = 0
    if os.path.exists(args.output_path):
        try:
            with open(args.output_path, "r", encoding="utf-8") as f:
                processed_count = len(f.readlines())
            print(f"发现已处理 {processed_count} 个问题，将从第 {processed_count+1} 个继续")
        except:
            processed_count = 0
    
    # 加载模型和分词器
    print(f"加载模型: {args.model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        
        # 获取</think>标记的ID
        think_token_id = get_think_token_id(tokenizer)
        print(f"获取</think>标记的ID: {think_token_id}")
        
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    # 生成解答
    results = []
    
    # 分批处理
    start_time = time.time()
    for batch_start in range(0, len(problems), args.batch_size):
        # 跳过已处理的样本
        if batch_start < processed_count:
            continue
            
        batch_end = min(batch_start + args.batch_size, len(problems))
        batch_problems = problems[batch_start:batch_end]
        
        batch_messages = [[{"role": "user", "content": problem}] for problem in batch_problems]
        batch_texts = [
            tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=False
            ) for messages in batch_messages
        ]
        
        # 处理每个样本
        for i, (problem, text) in enumerate(zip(batch_problems, batch_texts)):
            global_idx = batch_start + i
            print(f"处理问题 {global_idx + 1}/{len(problems)}")
            
            retry_count = 0
            success = False
            
            while retry_count < args.retry_count and not success:
                try:
                    # 准备输入
                    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
                    
                    # 生成回答
                    with torch.no_grad():
                        generated_ids = model.generate(
                            **model_inputs,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.8,
                            top_k=20
                        )
                    
                    # 提取生成的内容
                    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
                    
                    # 尝试分离思考内容和最终答案
                    try:
                        # </think>标记的索引位置
                        index = len(output_ids) - output_ids[::-1].index(think_token_id)
                    except ValueError:
                        print(f"未找到</think>标记，将完整输出作为解答")
                        index = 0
                    
                    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
                    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
                    
                    # 保存结果
                    result = {
                        "problem_idx": global_idx,  # 全局索引
                        "problem": problem,
                        "thinking": thinking_content,
                        "solution": content,
                    }
                    results.append(result)
                    
                    # 实时保存每个结果
                    save_results(result, args.output_path)
                    
                    success = True
                    
                except Exception as e:
                    retry_count += 1
                    print(f"处理失败 ({retry_count}/{args.retry_count}): {e}")
                    time.sleep(1)  # 等待一秒后重试
            
            if not success:
                print(f"跳过问题 {global_idx + 1} - 多次尝试后失败")
                # 记录失败的样本
                with open(f"{args.output_path}.failed", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"problem_idx": global_idx, "problem": problem, "error": "多次尝试后失败"}, ensure_ascii=False) + "\n")
            
            # 打印进度
            elapsed_time = time.time() - start_time
            avg_time = elapsed_time / (global_idx - batch_start + 1) if global_idx >= batch_start else 0
            remaining = (len(problems) - global_idx - 1) * avg_time if avg_time > 0 else 0
            print(f"进度: {global_idx + 1}/{len(problems)}, 已用时间: {elapsed_time:.1f}s, 平均每样本: {avg_time:.1f}s, 预计剩余时间: {remaining:.1f}s")
        
        # 每批次结束后清空GPU缓存
        torch.cuda.empty_cache()
    
    print(f"处理完成，共 {len(results)} 个结果已保存到 {args.output_path}")

if __name__ == "__main__":
    main() 