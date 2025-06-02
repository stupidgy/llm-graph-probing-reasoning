import json
import os
import argparse
from collections import defaultdict
import markdown
import re
from html import escape

def load_jsonl(file_path):
    """加载JSONL文件"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        print(f"从 {file_path} 读取了 {len(data)} 条记录")
        return data
    except Exception as e:
        print(f"加载 {file_path} 时出错: {e}")
        return []

def process_directory(dir_path, output_file, remove_problem_idx=False, sort_key='problem_idx', remove_duplicates=True):
    """处理指定目录中的所有JSONL文件"""
    all_data = []
    problem_set = set()  # 用于去重
    
    # 获取目录中的所有JSONL文件
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.jsonl')]
    print(f"在 {dir_path} 中发现 {len(files)} 个JSONL文件")
    
    # 读取所有文件
    for file_path in files:
        if os.path.exists(file_path):
            data = load_jsonl(file_path)
            
            # 记录文件来源
            file_name = os.path.basename(file_path)
            for item in data:
                item['source_file'] = file_name
                
                # 如果需要，移除problem_idx
                if remove_problem_idx and 'problem_idx' in item:
                    del item['problem_idx']
            
            # 去重处理
            if remove_duplicates:
                unique_data = []
                for item in data:
                    problem = item.get('problem', '')
                    if problem and problem not in problem_set:
                        problem_set.add(problem)
                        unique_data.append(item)
                    elif not problem:
                        unique_data.append(item)  # 如果没有problem字段，保留该条目
                
                print(f"从 {file_path} 中去除了 {len(data) - len(unique_data)} 条重复记录")
                all_data.extend(unique_data)
            else:
                all_data.extend(data)
        else:
            print(f"文件不存在: {file_path}")
    
    # 排序（如果需要）
    if sort_key and len(all_data) > 0 and sort_key in all_data[0]:
        all_data.sort(key=lambda x: x.get(sort_key, 0))
        print(f"已根据 {sort_key} 对 {len(all_data)} 条记录进行排序")
    
    # 写入合并后的文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"已将 {len(all_data)} 条记录合并写入到 {output_file}")
    return all_data

def find_common_problems(data1, data2, output_file1, output_file2):
    """找出两个数据集中共有的问题，并分别保存"""
    # 按问题文本分组
    problems_data1 = {}
    for item in data1:
        problem = item.get('problem', '')
        if problem:
            problems_data1[problem] = item
    
    # 找出共有问题
    common_data1 = []  # 来自数据集1的共有问题
    common_data2 = []  # 来自数据集2的共有问题
    common_problems = set()  # 用于记录共有问题
    
    # 先找出数据集1中在数据集2中也存在的问题
    for item in data1:
        problem = item.get('problem', '')
        if problem:
            for item2 in data2:
                if item2.get('problem', '') == problem:
                    common_data1.append(item)
                    common_problems.add(problem)
                    break
    
    # 找出数据集2中的共有问题版本
    for item in data2:
        problem = item.get('problem', '')
        if problem in common_problems:
            common_data2.append(item)
    
    # 写入共有问题文件（数据集1版本）
    with open(output_file1, 'w', encoding='utf-8') as f:
        for item in common_data1:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 写入共有问题文件（数据集2版本）
    with open(output_file2, 'w', encoding='utf-8') as f:
        for item in common_data2:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"找到 {len(common_problems)} 个共有问题")
    print(f"已将 {len(common_data1)} 条记录写入到 {output_file1}")
    print(f"已将 {len(common_data2)} 条记录写入到 {output_file2}")
    
    return common_data1, common_data2

def analyze_data(data):
    """分析数据，打印统计信息"""
    if not data:
        print("没有数据可分析")
        return
    
    # 计算问题数量
    problem_count = len(data)
    
    # 计算有thinking的问题数量
    with_thinking = sum(1 for item in data if item.get('thinking', '').strip())
    
    # 计算有solution的问题数量
    with_solution = sum(1 for item in data if item.get('solution', '').strip())
    
    # 计算thinking和solution平均长度
    thinking_lengths = [len(item.get('thinking', '')) for item in data]
    solution_lengths = [len(item.get('solution', '')) for item in data]
    
    avg_thinking_length = sum(thinking_lengths) / len(thinking_lengths) if thinking_lengths else 0
    avg_solution_length = sum(solution_lengths) / len(solution_lengths) if solution_lengths else 0
    
    print("\n数据分析:")
    print(f"总问题数量: {problem_count}")
    print(f"有思考过程的问题数量: {with_thinking} ({with_thinking/problem_count*100:.2f}%)")
    print(f"有解答的问题数量: {with_solution} ({with_solution/problem_count*100:.2f}%)")
    print(f"思考过程平均长度: {avg_thinking_length:.2f} 字符")
    print(f"解答平均长度: {avg_solution_length:.2f} 字符")
    
    # 按源文件分组统计
    by_source = defaultdict(list)
    for item in data:
        source = item.get('source_file', '未知')
        by_source[source].append(item)
    
    print("\n按源文件分组统计:")
    for source, items in by_source.items():
        with_think = sum(1 for item in items if item.get('thinking', '').strip())
        with_sol = sum(1 for item in items if item.get('solution', '').strip())
        
        print(f"源文件 {source}:")
        print(f"  问题数量: {len(items)}")
        print(f"  有思考过程的问题: {with_think} ({with_think/len(items)*100:.2f}%)")
        print(f"  有解答的问题: {with_sol} ({with_sol/len(items)*100:.2f}%)")

def compare_common_data(think_data, nothink_data):
    """比较共有问题在两个数据集中的情况"""
    if not think_data or not nothink_data:
        print("没有足够的共有问题数据可比较")
        return
    
    # 按问题文本整理数据
    think_dict = {item.get('problem', ''): item for item in think_data}
    nothink_dict = {item.get('problem', ''): item for item in nothink_data}
    
    # 找出共有问题
    common_problems = set(think_dict.keys()) & set(nothink_dict.keys())
    
    print(f"\n共有问题比较分析 (共 {len(common_problems)} 个):")
    
    # 统计思考过程长度与解答长度的差异
    total_think_thinking_length = 0
    total_nothink_thinking_length = 0
    total_think_solution_length = 0
    total_nothink_solution_length = 0
    
    for problem in common_problems:
        think_item = think_dict[problem]
        nothink_item = nothink_dict[problem]
        
        total_think_thinking_length += len(think_item.get('thinking', ''))
        total_nothink_thinking_length += len(nothink_item.get('thinking', ''))
        total_think_solution_length += len(think_item.get('solution', ''))
        total_nothink_solution_length += len(nothink_item.get('solution', ''))
    
    avg_think_thinking_length = total_think_thinking_length / len(common_problems)
    avg_nothink_thinking_length = total_nothink_thinking_length / len(common_problems)
    avg_think_solution_length = total_think_solution_length / len(common_problems)
    avg_nothink_solution_length = total_nothink_solution_length / len(common_problems)
    
    print(f"思考数据集中思考过程平均长度: {avg_think_thinking_length:.2f} 字符")
    print(f"无思考数据集中思考过程平均长度: {avg_nothink_thinking_length:.2f} 字符")
    print(f"思考数据集中解答平均长度: {avg_think_solution_length:.2f} 字符")
    print(f"无思考数据集中解答平均长度: {avg_nothink_solution_length:.2f} 字符")

def extract_model_info(file_name):
    """从文件名中提取模型信息"""
    model_info = {}
    
    # 提取rank信息
    if file_name.startswith('rank'):
        parts = file_name.split('_', 1)
        if len(parts) > 0:
            model_info['rank'] = parts[0]
    
    # 提取模型名称
    if 'qwen3' in file_name.lower():
        model_info['model'] = 'qwen3'
    
    # 提取是否包含thinking
    if 'nothink' in file_name.lower():
        model_info['thinking'] = False
    else:
        model_info['thinking'] = True
    
    return model_info

def main():
    parser = argparse.ArgumentParser(description="处理JSONL文件")
    parser.add_argument("--outputs_dir", type=str, default="outputs", help="outputs目录路径")
    parser.add_argument("--outputs_nothink_dir", type=str, default="outputs_nothink", help="outputs_nothink目录路径")
    parser.add_argument("--think_merged_file", type=str, default="think_merged.jsonl", help="有思考过程的数据合并后的输出文件")
    parser.add_argument("--nothink_merged_file", type=str, default="nothink_merged.jsonl", help="无思考过程的数据合并后的输出文件")
    parser.add_argument("--think_common_file", type=str, default="think_common.jsonl", help="有思考过程的共有问题输出文件")
    parser.add_argument("--nothink_common_file", type=str, default="nothink_common.jsonl", help="无思考过程的共有问题输出文件")
    parser.add_argument("--analyze", action="store_true", help="分析合并后的数据")
    parser.add_argument("--remove_problem_idx", action="store_true", help="移除problem_idx字段")
    args = parser.parse_args()
    
    # 步骤1：处理outputs目录（有思考过程的数据）
    print("步骤1：处理outputs目录中的数据")
    think_data = process_directory(
        args.outputs_dir, 
        args.think_merged_file,
        remove_problem_idx=args.remove_problem_idx
    )
    
    # 步骤2：处理outputs_nothink目录（无思考过程的数据）
    print("\n步骤2：处理outputs_nothink目录中的数据")
    nothink_data = process_directory(
        args.outputs_nothink_dir, 
        args.nothink_merged_file,
        remove_problem_idx=args.remove_problem_idx
    )
    
    # 分析数据
    if args.analyze:
        print("\n分析有思考过程的数据:")
        analyze_data(think_data)
        
        print("\n分析无思考过程的数据:")
        analyze_data(nothink_data)
    
    # 步骤3：找出共有问题，并保存两个版本
    print("\n步骤3：找出共有问题")
    think_common, nothink_common = find_common_problems(
        think_data,
        nothink_data,
        args.think_common_file,
        args.nothink_common_file
    )
    
    # 比较共有问题在两个数据集中的情况
    if args.analyze:
        compare_common_data(think_common, nothink_common)

if __name__ == "__main__":
    main() 