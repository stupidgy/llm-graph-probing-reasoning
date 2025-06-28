import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import functional as F
import copy
from typing import Dict, List, Tuple, Optional, Union


class NeuralInterventionController:
    """神经干预控制器，用于对语言模型的特定层和特定维度进行干预"""
    
    def __init__(self, model_name_or_path: str, device: str = "cuda", gpu_id: int = 0):
        """
        初始化干预控制器
        
        Args:
            model_name_or_path: 模型路径或名称
            device: 设备类型（cuda或cpu）
            gpu_id: GPU设备ID（当device为cuda时生效）
        """
        # 设置具体的设备
        if device == "cuda" and torch.cuda.is_available():
            self.device = f"cuda:{gpu_id}"
            # 设置默认GPU设备
            torch.cuda.set_device(gpu_id)
            print(f"使用GPU设备: {self.device}")
            print(f"GPU设备名称: {torch.cuda.get_device_name(gpu_id)}")
            print(f"GPU总内存: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.2f} GB")
        else:
            self.device = "cpu"
            print(f"使用CPU设备")
        
        self.model_name_or_path = model_name_or_path
        
        # 加载模型和分词器
        print(f"正在加载模型: {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map=self.device
        ).to(self.device)
        
        # 设置padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        
        # 存储激活值的字典
        self.activations = {}
        # 存储hook的列表
        self.hooks = []
        # 干预配置
        self.interventions = {}
        
        # 位置感知干预相关
        self.prompt_length = 0  # 记录prompt的长度
        self.intervention_start_pos = 0  # 干预开始位置
        self.current_generation_step = 0  # 当前生成步骤
        self.last_intervened_length = 0  # 上次干预时的序列长度，用于避免重复干预
        
        print(f"模型层数: {len(self.model.model.layers)}")
        print(f"隐藏状态维度: {self.model.config.hidden_size}")
    
    def register_activation_hook(self, layer_idx: int):
        """注册激活值获取钩子"""
        def hook_fn(module, input, output):
            # output[0] 是hidden states
            self.activations[f'layer_{layer_idx}'] = output[0].detach().clone()
        
        # 获取指定层
        layer = self.model.model.layers[layer_idx]
        hook = layer.register_forward_hook(hook_fn)
        self.hooks.append(hook)
        return hook
    
    def register_intervention_hook(self, layer_idx: int, intervention_fn):
        """注册干预钩子"""
        def hook_fn(module, input, output):
            # 应用干预函数，传入位置信息
            modified_output = intervention_fn(output[0], self.prompt_length, self.intervention_start_pos)
            # 返回修改后的输出，保持原始元组结构
            return (modified_output,) + output[1:] if len(output) > 1 else (modified_output,)
        
        layer = self.model.model.layers[layer_idx]
        hook = layer.register_forward_hook(hook_fn)
        self.hooks.append(hook)
        return hook
    
    def clear_hooks(self):
        """清除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.activations.clear()
    
    def set_dimension_intervention(self, layer_idx: int, dimensions: List[int], 
                                 intervention_type: str = "gaussian_replace", 
                                 intervention_value: float = 0.0,
                                 scale_factor: float = 1.0,
                                 gaussian_mean: float = 0.0,
                                 gaussian_std: float = 1.0,
                                 intervene_on_prompt: bool = False):
        """
        设置特定维度的干预
        
        Args:
            layer_idx: 目标层索引
            dimensions: 要干预的维度列表
            intervention_type: 干预类型 ("gaussian_replace", "gaussian_noise", "zero", "constant", "scale", "invert")
            intervention_value: 干预值（对于constant类型）
            scale_factor: 缩放因子（对于scale类型）
            gaussian_mean: 高斯分布均值（对于gaussian_replace类型）
            gaussian_std: 高斯分布标准差（对于gaussian_replace类型）
            intervene_on_prompt: 是否对prompt部分也进行干预（默认False，只对生成部分干预）
        """
        def intervention_fn(hidden_states, prompt_length, intervention_start_pos):
            """干预函数，只对新生成的token进行干预，避免重复干预"""
            modified_states = hidden_states.clone()
            seq_len = hidden_states.shape[1]
            
            # 确定干预的位置范围
            if intervene_on_prompt:
                # 对整个序列进行干预
                start_pos = 0
                end_pos = seq_len
            else:
                # 只对新生成的token进行干预，避免重复干预
                start_pos = max(self.last_intervened_length, prompt_length)
                end_pos = seq_len
                # 更新已干预的长度
                self.last_intervened_length = seq_len
            
            # 如果没有需要干预的位置，直接返回
            if start_pos >= end_pos:
                return modified_states
            
            for dim in dimensions:
                if intervention_type == "gaussian_replace":
                    # 用高斯随机分布完全替代指定维度
                    shape = modified_states[:, start_pos:end_pos, dim].shape
                    gaussian_values = torch.normal(
                        mean=gaussian_mean, 
                        std=gaussian_std, 
                        size=shape, 
                        device=modified_states.device,
                        dtype=modified_states.dtype
                    )
                    modified_states[:, start_pos:end_pos, dim] = gaussian_values
                elif intervention_type == "gaussian_noise":
                    # 在原有激活值基础上添加高斯噪声
                    shape = modified_states[:, start_pos:end_pos, dim].shape
                    noise = torch.normal(
                        mean=gaussian_mean, 
                        std=gaussian_std, 
                        size=shape, 
                        device=modified_states.device,
                        dtype=modified_states.dtype
                    )
                    modified_states[:, start_pos:end_pos, dim] += noise
                elif intervention_type == "zero":
                    # 将指定维度置零
                    modified_states[:, start_pos:end_pos, dim] = 0.0
                elif intervention_type == "constant":
                    # 将指定维度设为常数
                    modified_states[:, start_pos:end_pos, dim] = intervention_value
                elif intervention_type == "scale":
                    # 缩放指定维度
                    modified_states[:, start_pos:end_pos, dim] *= scale_factor
                elif intervention_type == "noise":
                    # 添加噪声（兼容旧版本）
                    noise = torch.randn_like(modified_states[:, start_pos:end_pos, dim]) * intervention_value
                    modified_states[:, start_pos:end_pos, dim] += noise
                elif intervention_type == "invert":
                    # 反转激活值
                    modified_states[:, start_pos:end_pos, dim] = -modified_states[:, start_pos:end_pos, dim]
            
            return modified_states
        
        # 注册干预钩子
        self.register_intervention_hook(layer_idx, intervention_fn)
        
        # 记录干预配置
        self.interventions[layer_idx] = {
            'dimensions': dimensions,
            'type': intervention_type,
            'value': intervention_value,
            'scale_factor': scale_factor,
            'gaussian_mean': gaussian_mean,
            'gaussian_std': gaussian_std,
            'intervene_on_prompt': intervene_on_prompt
        }
    
    def generate_with_dual_mode_intervention(self, prompt: str, 
                                            # 通用参数
                                            max_new_tokens: int = 32768,
                                            target_layer: int = None, 
                                            target_dimensions: List[int] = None,
                                            # NoThink模式参数
                                            nothink_temperature: float = 0.7,
                                            nothink_top_k: int = 20,
                                            nothink_top_p: float = 0.8,
                                            nothink_do_sample: bool = True,
                                            # Think模式参数
                                            think_temperature: float = 0.6,
                                            think_top_k: int = 20,
                                            think_top_p: float = 0.95,
                                            think_do_sample: bool = True) -> Dict:
        """
        同时进行think和nothink模式的干预生成，并统计激活值，支持为两种模式设置不同的生成参数
        
        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成token数（两种模式共用）
            target_layer: 目标层（用于统计激活值）
            target_dimensions: 目标维度（用于统计激活值）
            
            # NoThink模式参数
            nothink_temperature: NoThink模式的温度参数
            nothink_top_k: NoThink模式的top-k sampling参数
            nothink_top_p: NoThink模式的nucleus sampling参数
            nothink_do_sample: NoThink模式是否使用采样
            
            # Think模式参数  
            think_temperature: Think模式的温度参数
            think_top_k: Think模式的top-k sampling参数
            think_top_p: Think模式的nucleus sampling参数
            think_do_sample: Think模式是否使用采样
        
        Returns:
            包含两种模式下原始输出、干预后输出和激活值统计的字典
        """
        results = {}
        
        # 获取</think>标记的ID
        think_token_id = self._get_think_token_id()
        
        # 定义两种模式的配置
        mode_configs = {
            "nothink": {
                "think_mode": False,
                "generation_kwargs": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": nothink_temperature,
                    "top_k": nothink_top_k if nothink_top_k is not None else None,
                    "top_p": nothink_top_p,
                    "do_sample": nothink_do_sample,
                    "pad_token_id": self.tokenizer.eos_token_id,
                }
            },
            "think": {
                "think_mode": True,
                "generation_kwargs": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": think_temperature,
                    "top_k": think_top_k if think_top_k is not None else None,
                    "top_p": think_top_p,
                    "do_sample": think_do_sample,
                    "pad_token_id": self.tokenizer.eos_token_id,
                }
            }
        }
        
        # 为两种模式进行实验
        for mode_name, config in mode_configs.items():
            think_mode = config["think_mode"]
            generation_kwargs = {k: v for k, v in config["generation_kwargs"].items() if v is not None}
            
            print(f"\n处理{mode_name}模式...")
            print(f"  生成参数: temperature={generation_kwargs.get('temperature')}, "
                  f"top_k={generation_kwargs.get('top_k')}, "
                  f"top_p={generation_kwargs.get('top_p')}, "
                  f"do_sample={generation_kwargs.get('do_sample')}")
            
            # 使用对应模式的模板
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=think_mode
            )
            
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt", padding=True).to(self.device)
            
            # 记录prompt长度
            self.prompt_length = inputs['input_ids'].shape[1]
            self.intervention_start_pos = self.prompt_length
            self.last_intervened_length = self.prompt_length
            
            # 1. 原始生成（用于统计激活值）
            print(f"  {mode_name}模式原始生成...")
            self.clear_hooks()
            
            # 如果指定了目标层和维度，注册激活值钩子进行统计
            activation_stats = None
            if target_layer is not None and target_dimensions is not None:
                # 为每个维度单独存储激活值
                activation_values_by_dim = {dim: [] for dim in target_dimensions}
                hook_call_count = 0  # 添加钩子调用计数
                
                def stats_hook_fn(module, input, output):
                    nonlocal hook_call_count
                    hook_call_count += 1
                    
                    # 获取目标维度的激活值
                    hidden_states = output[0]
                    seq_len = hidden_states.shape[1]
                    
                    print(f"    钩子被调用 #{hook_call_count}: seq_len={seq_len}, prompt_length={self.prompt_length}")
                    
                    # 直接收集所有激活值，不进行任何过滤
                    print(f"    正在收集第 {hook_call_count} 个token的激活值")
                    
                    for dim_idx, dim in enumerate(target_dimensions):
                        # 对于seq_len=1的情况，取第0个位置；对于seq_len>1的情况，取最后一个位置
                        if seq_len == 1:
                            token_activation = hidden_states[:, 0, dim]
                        else:
                            token_activation = hidden_states[:, -1, dim]  # 取最后一个位置
                        
                        activation_value = token_activation.flatten().detach().cpu().numpy()[0]
                        activation_values_by_dim[dim].append(activation_value)
                        print(f"      维度 {dim}: 激活值 = {activation_value:.6f}")
                    
                    total_collected = sum(len(values) for values in activation_values_by_dim.values())
                    print(f"    已收集激活值总数: {total_collected}")
                
                # 注册统计钩子
                layer = self.model.model.layers[target_layer]
                stats_hook = layer.register_forward_hook(stats_hook_fn)
                print(f"  {mode_name}模式: 已注册激活值统计钩子在第{target_layer}层")
            
            with torch.no_grad():
                original_outputs = self.model.generate(**inputs, **generation_kwargs)
            
            # 计算激活值统计
            if target_layer is not None and target_dimensions is not None:
                stats_hook.remove()
                
                # 详细的调试信息
                original_output_ids = original_outputs[0][len(inputs.input_ids[0]):].tolist()
                original_text = self.tokenizer.decode(original_outputs[0], skip_special_tokens=True)
                generated_text = self.tokenizer.decode(original_output_ids, skip_special_tokens=False)
                
                total_collected = sum(len(values) for values in activation_values_by_dim.values())
                
                print(f"  {mode_name}模式调试信息:")
                print(f"    钩子总调用次数: {hook_call_count}")
                print(f"    收集到的激活值总数: {total_collected}")
                print(f"    目标维度数量: {len(target_dimensions)}")
                print(f"    每个维度收集的token数: {len(list(activation_values_by_dim.values())[0]) if activation_values_by_dim else 0}")
                print(f"    实际生成的token ID数量: {len(original_output_ids)}")
                print(f"    生成的文本长度: {len(generated_text)}")
                print(f"    prompt长度: {self.prompt_length}")
                
                # 为每个维度分别统计
                for dim in target_dimensions:
                    values = activation_values_by_dim[dim]
                    if values:
                        print(f"    维度 {dim} 统计: count={len(values)}, mean={np.mean(values):.6f}, std={np.std(values):.6f}")
                
                if activation_values_by_dim and any(values for values in activation_values_by_dim.values()):
                    activation_stats = {
                        'by_dimension': {
                            dim: {
                                'mean': float(np.mean(values)) if values else 0.0,
                                'std': float(np.std(values)) if values else 0.0,
                                'variance': float(np.var(values)) if values else 0.0,
                                'min': float(np.min(values)) if values else 0.0,
                                'max': float(np.max(values)) if values else 0.0,
                                'count': len(values)
                            }
                            for dim, values in activation_values_by_dim.items()
                        },
                        'overall': {
                            'total_collected': total_collected,
                            'tokens_per_dimension': len(list(activation_values_by_dim.values())[0]) if activation_values_by_dim else 0,
                            'actual_generated_tokens': len(original_output_ids)
                        }
                    }
                    print(f"  {mode_name}模式激活值统计完成")
                else:
                    print(f"  {mode_name}模式: 未收集到激活值")
                    activation_stats = None
            
            # 提取生成的内容
            original_output_ids = original_outputs[0][len(inputs.input_ids[0]):].tolist()
            original_text = self.tokenizer.decode(original_outputs[0], skip_special_tokens=True)
            
            # 分离thinking和solution
            original_thinking, original_solution = self._separate_thinking_and_solution(
                original_output_ids, think_token_id, think_mode
            )
            
            # 2. 干预生成
            print(f"  {mode_name}模式干预生成...")
            # 重置干预状态
            self.last_intervened_length = self.prompt_length
            
            # 重新注册干预钩子
            for layer_idx, config_item in self.interventions.items():
                self.set_dimension_intervention(
                    layer_idx, 
                    config_item['dimensions'], 
                    config_item['type'],
                    config_item['value'],
                    config_item['scale_factor'],
                    config_item.get('gaussian_mean', 0.0),
                    config_item.get('gaussian_std', 1.0),
                    config_item.get('intervene_on_prompt', False)
                )
            
            with torch.no_grad():
                intervention_outputs = self.model.generate(**inputs, **generation_kwargs)
            
            # 提取干预后的生成内容
            intervention_output_ids = intervention_outputs[0][len(inputs.input_ids[0]):].tolist()
            intervention_text = self.tokenizer.decode(intervention_outputs[0], skip_special_tokens=True)
            
            # 分离thinking和solution
            intervention_thinking, intervention_solution = self._separate_thinking_and_solution(
                intervention_output_ids, think_token_id, think_mode
            )
            
            # 保存该模式的结果
            results[mode_name] = {
                'formatted_prompt': formatted_prompt,
                'original_text': original_text,
                'intervention_text': intervention_text,
                'original_thinking': original_thinking,
                'original_solution': original_solution,
                'intervention_thinking': intervention_thinking,
                'intervention_solution': intervention_solution,
                'activation_stats': activation_stats,
                'prompt_length': self.prompt_length,
                'generation_config': generation_kwargs
            }
        
        # 返回完整结果
        return {
            'prompt': prompt,
            'nothink_mode': results['nothink'],
            'think_mode': results['think'],
            'interventions': copy.deepcopy(self.interventions),
            'generation_configs': {
                'nothink': mode_configs['nothink']['generation_kwargs'],
                'think': mode_configs['think']['generation_kwargs']
            },
            'activation_analysis': {
                'nothink_mode': results['nothink']['activation_stats'],
                'think_mode': results['think']['activation_stats']
            }
        }
    
    def _get_think_token_id(self):
        """获取</think>的token ID"""
        try:
            think_token = "</think>"
            think_token_id = self.tokenizer.convert_tokens_to_ids(think_token)
            if think_token_id == self.tokenizer.unk_token_id:
                # 如果转换结果是unk token，尝试其他方式
                think_token_id = None
                
                # 方法1: 尝试编码整个标记并获取最后一个token
                tokens = self.tokenizer.tokenize(think_token)
                if tokens and tokens[-1] != self.tokenizer.unk_token:
                    think_token_id = self.tokenizer.convert_tokens_to_ids(tokens[-1])
                
                # 如果都找不到，使用默认值
                if think_token_id is None or think_token_id == self.tokenizer.unk_token_id:
                    # Qwen3的</think>标记ID可能是151668
                    print("无法确定</think>标记ID，使用默认值151668")
                    think_token_id = 151668
            
            return think_token_id
        except Exception as e:
            print(f"获取</think>标记ID失败: {e}")
            return 151668  # 默认值
    
    def _separate_thinking_and_solution(self, output_ids, think_token_id, think_mode):
        """分离thinking内容和solution内容"""
        if not think_mode:
            # nothink模式，整个输出都是solution
            solution = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            return "", solution
        
        try:
            # 寻找</think>标记的位置
            think_index = len(output_ids) - output_ids[::-1].index(think_token_id)
        except ValueError:
            # 未找到</think>标记，将完整输出作为solution
            solution = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            return "", solution
        
        # 分离thinking和solution
        thinking_content = self.tokenizer.decode(output_ids[:think_index], skip_special_tokens=True).strip()
        solution_content = self.tokenizer.decode(output_ids[think_index:], skip_special_tokens=True).strip()
        
        return thinking_content, solution_content

def run_intervention_experiment(model_path: str, 
                              prompts: List[str],
                              target_layer: int,
                              target_dimensions: List[int],
                              intervention_types: List[str] = ["gaussian_replace"],
                              scale_factors: List[float] = [0.5, 2.0],
                              gaussian_params: List[Dict] = None,
                              output_dir: str = "intervention_results",
                              # 通用参数
                              max_new_tokens: int = 32768,
                              # NoThink模式参数
                              nothink_temperature: float = 0.7,
                              nothink_top_k: int = 20,
                              nothink_top_p: float = 0.8,
                              nothink_do_sample: bool = True,
                              # Think模式参数
                              think_temperature: float = 0.6,
                              think_top_k: int = 20,
                              think_top_p: float = 0.95,
                              think_do_sample: bool = True,
                              device: str = "cuda",
                              gpu_id: int = 0):
    """
    运行干预实验，支持为think和nothink模式设置不同的生成参数
    
    Args:
        model_path: 模型路径
        prompts: 测试提示列表
        target_layer: 目标层
        target_dimensions: 目标维度列表
        intervention_types: 干预类型列表
        scale_factors: 缩放因子列表（仅对scale类型有效）
        gaussian_params: 高斯分布参数列表 [{"mean": 0.0, "std": 1.0}, ...]
        output_dir: 输出目录
        
        # 通用参数
        max_new_tokens: 最大生成token数（两种模式共用）
        
        # NoThink模式参数
        nothink_temperature: NoThink模式的温度参数
        nothink_top_k: NoThink模式的top-k sampling参数
        nothink_top_p: NoThink模式的nucleus sampling参数
        nothink_do_sample: NoThink模式是否使用采样
        
        # Think模式参数
        think_temperature: Think模式的温度参数
        think_top_k: Think模式的top-k sampling参数
        think_top_p: Think模式的nucleus sampling参数
        think_do_sample: Think模式是否使用采样
        
        device: 设备类型
        gpu_id: GPU设备ID
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 输出调试信息
    print(f"当前工作目录: {os.getcwd()}")
    print(f"输出目录: {os.path.abspath(output_dir)}")
    
    # 默认高斯参数
    if gaussian_params is None:
        gaussian_params = [
            {"mean": 500, "std": 0}
        ]
    
    # 初始化控制器
    controller = NeuralInterventionController(model_path, device, gpu_id)
    
    all_results = []
    
    print(f"双模式生成配置:")
    print(f"  NoThink模式: temperature={nothink_temperature}, top_k={nothink_top_k}, top_p={nothink_top_p}, do_sample={nothink_do_sample}")
    print(f"  Think模式: temperature={think_temperature}, top_k={think_top_k}, top_p={think_top_p}, do_sample={think_do_sample}")
    print(f"  最大生成token数: {max_new_tokens}")
    
    for prompt_idx, prompt in enumerate(prompts):
        print(f"\n处理提示 {prompt_idx + 1}/{len(prompts)}: {prompt[:50]}...")
        
        for intervention_type in intervention_types:
            if intervention_type == "scale":
                test_values = scale_factors
                param_type = "scale"
            elif intervention_type in ["gaussian_replace", "gaussian_noise"]:
                test_values = gaussian_params
                param_type = "gaussian"
            else:
                test_values = [0.0]  # 对于其他类型，使用默认值
                param_type = "default"
            
            for value in test_values:
                if param_type == "gaussian":
                    print(f"  干预类型: {intervention_type}, 高斯参数: mean={value['mean']}, std={value['std']}")
                else:
                    print(f"  干预类型: {intervention_type}, 值: {value}")
                
                # 清除之前的干预
                controller.clear_hooks()
                controller.interventions.clear()
                
                # 设置新的干预
                if intervention_type == "scale":
                    controller.set_dimension_intervention(
                        target_layer, target_dimensions, intervention_type, 
                        scale_factor=value
                    )
                elif intervention_type in ["gaussian_replace", "gaussian_noise"]:
                    controller.set_dimension_intervention(
                        target_layer, target_dimensions, intervention_type,
                        gaussian_mean=value["mean"],
                        gaussian_std=value["std"]
                    )
                else:
                    controller.set_dimension_intervention(
                        target_layer, target_dimensions, intervention_type, 
                        intervention_value=value
                    )
                
                # 生成结果，传递分别的模式参数
                result = controller.generate_with_dual_mode_intervention(
                    prompt, 
                    max_new_tokens=max_new_tokens,
                    target_layer=target_layer,
                    target_dimensions=target_dimensions,
                    nothink_temperature=nothink_temperature,
                    nothink_top_k=nothink_top_k,
                    nothink_top_p=nothink_top_p,
                    nothink_do_sample=nothink_do_sample,
                    think_temperature=think_temperature,
                    think_top_k=think_top_k,
                    think_top_p=think_top_p,
                    think_do_sample=think_do_sample
                )
                result['prompt_idx'] = prompt_idx
                if param_type == "gaussian":
                    result['intervention_config'] = {
                        'layer': target_layer,
                        'dimensions': target_dimensions,
                        'type': intervention_type,
                        'gaussian_mean': value["mean"],
                        'gaussian_std': value["std"]
                    }
                else:
                    result['intervention_config'] = {
                        'layer': target_layer,
                        'dimensions': target_dimensions,
                        'type': intervention_type,
                        'value': value
                    }
                
                all_results.append(result)
                
    
    # 保存结果
    output_file = os.path.join(output_dir, f"dual_mode_intervention_layer_{target_layer}_dims_{'-'.join(map(str, target_dimensions))}.json")
    
    # 确保输出文件的目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 转换tensor为可序列化格式
    serializable_results = []
    for result in all_results:
        serializable_result = copy.deepcopy(result)
        # 激活值统计已经是基本数据类型，可以直接序列化，不需要删除
        # 只移除可能存在的torch tensor对象（如果有的话）
        serializable_results.append(serializable_result)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {output_file}")
    
    # 生成双模式分析报告
    generate_dual_mode_analysis_report(all_results, output_dir, target_layer, target_dimensions)
    
    return all_results


def generate_dual_mode_analysis_report(results: List[Dict], output_dir: str, 
                           target_layer: int, target_dimensions: List[int]):
    """生成双模式分析报告（Markdown格式）"""
    report_file = os.path.join(output_dir, f"dual_mode_analysis_report_layer_{target_layer}_dims_{'-'.join(map(str, target_dimensions))}.md")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 神经干预双模式实验分析报告\n\n")
        f.write(f"## 实验配置\n\n")
        f.write(f"- **目标层**: {target_layer}\n")
        f.write(f"- **目标维度**: {target_dimensions}\n")
        f.write(f"- **总实验数**: {len(results)}\n\n")
        
        # 统计不同干预类型的结果
        intervention_types = {}
        for result in results:
            intervention_type = result['intervention_config']['type']
            if intervention_type not in intervention_types:
                intervention_types[intervention_type] = []
            intervention_types[intervention_type].append(result)
        
        for intervention_type, type_results in intervention_types.items():
            f.write(f"## {intervention_type} 干预类型结果\n\n")
            
            # 统计激活值变化（如果可用）
            if 'activation_analysis' in type_results[0] and type_results[0]['activation_analysis']:
                think_stats_list = []
                nothink_stats_list = []
                
                for result in type_results:
                    activation_analysis = result['activation_analysis']
                    if activation_analysis.get('think_mode'):
                        think_stats_list.append(activation_analysis['think_mode'])
                    if activation_analysis.get('nothink_mode'):
                        nothink_stats_list.append(activation_analysis['nothink_mode'])
                
                if think_stats_list and nothink_stats_list:
                    # 处理新的数据结构（分维度统计）
                    f.write(f"### 激活值统计\n\n")
                    
                    # 如果有分维度统计
                    if (think_stats_list[0] and 'by_dimension' in think_stats_list[0] and
                        nothink_stats_list[0] and 'by_dimension' in nothink_stats_list[0]):
                        
                        # 获取所有维度
                        dimensions = list(think_stats_list[0]['by_dimension'].keys())
                        
                        # 创建表格表头
                        f.write("| 维度 | 模式 | 平均均值 | 平均方差 | 平均样本数 |\n")
                        f.write("|------|------|----------|----------|------------|\n")
                        
                        for dim in dimensions:
                            # 计算每个维度的平均统计
                            think_means = [stats['by_dimension'][dim]['mean'] for stats in think_stats_list 
                                         if stats and 'by_dimension' in stats and dim in stats['by_dimension']]
                            think_vars = [stats['by_dimension'][dim]['variance'] for stats in think_stats_list 
                                        if stats and 'by_dimension' in stats and dim in stats['by_dimension']]
                            think_counts = [stats['by_dimension'][dim]['count'] for stats in think_stats_list 
                                          if stats and 'by_dimension' in stats and dim in stats['by_dimension']]
                            
                            nothink_means = [stats['by_dimension'][dim]['mean'] for stats in nothink_stats_list 
                                           if stats and 'by_dimension' in stats and dim in stats['by_dimension']]
                            nothink_vars = [stats['by_dimension'][dim]['variance'] for stats in nothink_stats_list 
                                          if stats and 'by_dimension' in stats and dim in stats['by_dimension']]
                            nothink_counts = [stats['by_dimension'][dim]['count'] for stats in nothink_stats_list 
                                            if stats and 'by_dimension' in stats and dim in stats['by_dimension']]
                            
                            if think_means and nothink_means:
                                avg_think_mean = sum(think_means) / len(think_means)
                                avg_think_var = sum(think_vars) / len(think_vars)
                                avg_think_count = sum(think_counts) / len(think_counts)
                                
                                avg_nothink_mean = sum(nothink_means) / len(nothink_means)
                                avg_nothink_var = sum(nothink_vars) / len(nothink_vars)
                                avg_nothink_count = sum(nothink_counts) / len(nothink_counts)

                                f.write(f"| {dim} | Think模式 | {avg_think_mean:.4f} | {avg_think_var:.4f} | {avg_think_count:.0f} |\n")
                                f.write(f"| {dim} | NoThink模式 | {avg_nothink_mean:.4f} | {avg_nothink_var:.4f} | {avg_nothink_count:.0f} |\n")
                        f.write("\n")
                    else:
                        f.write("**激活值统计**: 数据不完整\n\n")
                else:
                    f.write("**激活值统计**: 未收集到完整数据\n\n")
            else:
                f.write("**激活值统计**: 未启用统计功能\n\n")
            
            # 显示前3个例子
            f.write("### 实验示例\n\n")
            for i, result in enumerate(type_results[:3]):
                f.write(f"#### 例子 {i+1}\n\n")
                f.write(f"**提示**: {result['prompt']}\n\n")
                f.write(f"**干预配置**: {result['intervention_config']}\n\n")
                
                # NoThink模式结果
                f.write(f"##### NoThink模式\n\n")
                f.write(f"**原始回答** (长度: {len(result['nothink_mode']['original_solution'])})\n\n")
                f.write(f"{result['nothink_mode']['original_solution']}\n\n")
                f.write(f"**干预回答** (长度: {len(result['nothink_mode']['intervention_solution'])})\n\n")
                f.write(f"{result['nothink_mode']['intervention_solution']}\n\n")

                # Think模式结果
                f.write(f"##### Think模式\n\n")
                f.write(f"**原始思考过程** (长度: {len(result['think_mode']['original_thinking'])})\n\n")
                f.write(f"{result['think_mode']['original_thinking']}\n\n")
                f.write(f"**原始回答** (长度: {len(result['think_mode']['original_solution'])})\n\n")
                f.write(f"{result['think_mode']['original_solution']}\n\n")
                f.write(f"**干预思考过程** (长度: {len(result['think_mode']['intervention_thinking'])})\n\n")
                f.write(f"{result['think_mode']['intervention_thinking']}\n\n")
                f.write(f"**干预回答** (长度: {len(result['think_mode']['intervention_solution'])})\n\n")
                f.write(f"{result['think_mode']['intervention_solution']}\n\n")
                
                f.write(f"---\n\n")
    
    print(f"双模式分析报告已保存到: {report_file}")


def parse_args():
    parser = argparse.ArgumentParser(description='神经干预实验 - 支持双模式不同参数')
    parser.add_argument('--model_path', type=str, 
                        default='/data4/huguangyi/models/Qwen/Qwen3-0.6B',
                        help='模型路径')
    parser.add_argument('--target_layer', type=int, default=14,
                        help='目标层索引')
    parser.add_argument('--target_dimensions', type=str, default='16,18',
                        help='目标维度，用逗号分隔')
    parser.add_argument('--intervention_types', type=str, default='gaussian_replace',
                        help='干预类型，用逗号分隔')
    parser.add_argument('--prompts_file', type=str, default=None,
                        help='包含测试提示的文件路径')
    parser.add_argument('--output_dir', type=str, default='intervention_results',
                        help='输出目录')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')
    
    # GPU设备参数
    parser.add_argument('--gpu_id', type=int, default=5,
                        help='指定使用的GPU设备ID（默认：4）')
    
    # 通用生成配置参数
    parser.add_argument('--max_new_tokens', type=int, default=32768,
                        help='最大生成token数（两种模式共用）')
    
    # NoThink模式生成配置参数
    parser.add_argument('--nothink_temperature', type=float, default=0.7,
                        help='NoThink模式的温度参数，控制随机性')
    parser.add_argument('--nothink_top_k', type=int, default=20,
                        help='NoThink模式的top-k sampling参数')
    parser.add_argument('--nothink_top_p', type=float, default=0.8,
                        help='NoThink模式的nucleus sampling参数')
    parser.add_argument('--nothink_do_sample', action='store_true', default=True,
                        help='NoThink模式是否使用采样')
    parser.add_argument('--nothink_no_sample', dest='nothink_do_sample', action='store_false',
                        help='NoThink模式禁用采样')
    
    # Think模式生成配置参数
    parser.add_argument('--think_temperature', type=float, default=0.6,
                        help='Think模式的温度参数，控制随机性')
    parser.add_argument('--think_top_k', type=int, default=20,
                        help='Think模式的top-k sampling参数')
    parser.add_argument('--think_top_p', type=float, default=0.95,
                        help='Think模式的nucleus sampling参数')
    parser.add_argument('--think_do_sample', action='store_true', default=True,
                        help='Think模式是否使用采样')
    parser.add_argument('--think_no_sample', dest='think_do_sample', action='store_false',
                        help='Think模式禁用采样')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 解析目标维度
    target_dimensions = [int(d.strip()) for d in args.target_dimensions.split(',')]
    
    # 解析干预类型
    intervention_types = [t.strip() for t in args.intervention_types.split(',')]
    
    # 准备测试提示
    if args.prompts_file and os.path.exists(args.prompts_file):
        with open(args.prompts_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        # 默认测试提示
        prompts = [
            "The value of log base a certain number ( log 5625 ) is 2. What is that base?",
            "The value of log base a certain number ( log 5625 ) is 2. What is that base?",
            "The value of log base a certain number ( log 5625 ) is 2. What is that base?"
        ]
    
    print(f"将对模型 {args.model_path} 进行神经干预双模式实验")
    print(f"设备: {args.device}")
    if args.device == "cuda":
        print(f"GPU ID: {args.gpu_id}")
    print(f"目标层: {args.target_layer}")
    print(f"目标维度: {target_dimensions}")
    print(f"干预类型: {intervention_types}")
    print(f"测试提示数量: {len(prompts)}")
    print(f"最大生成token数: {args.max_new_tokens}")
    print(f"NoThink模式参数: temperature={args.nothink_temperature}, top_k={args.nothink_top_k}, top_p={args.nothink_top_p}, do_sample={args.nothink_do_sample}")
    print(f"Think模式参数: temperature={args.think_temperature}, top_k={args.think_top_k}, top_p={args.think_top_p}, do_sample={args.think_do_sample}")
    
    # 运行实验
    results = run_intervention_experiment(
        model_path=args.model_path,
        prompts=prompts,
        target_layer=args.target_layer,
        target_dimensions=target_dimensions,
        intervention_types=intervention_types,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        nothink_temperature=args.nothink_temperature,
        nothink_top_k=args.nothink_top_k,
        nothink_top_p=args.nothink_top_p,
        nothink_do_sample=args.nothink_do_sample,
        think_temperature=args.think_temperature,
        think_top_k=args.think_top_k,
        think_top_p=args.think_top_p,
        think_do_sample=args.think_do_sample,
        device=args.device,
        gpu_id=args.gpu_id
    )
    
    print(f"\n实验完成！共进行了 {len(results)} 次干预测试。")


if __name__ == "__main__":
    main() 