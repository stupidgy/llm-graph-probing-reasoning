import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.explain import Explainer, GNNExplainer, CaptumExplainer
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx

from graph_probing.model import GCNClassifier, GCNClassifierLinear
from graph_probing.utils import llm_model_num_nodes_map
from graph_probing.dataset import get_binary_classification_dataloader
from graph_probing.dataset import wrap_binary_data  # 导入单个数据处理函数

# 添加直接加载单个样本的功能
def load_single_sample(think_path, nothink_path, network_id, llm_layer, network_density, from_sparse_data=False):
    """直接加载单个样本，避免加载整个数据集"""
    # 确定样本来自哪个类别和路径
    if network_id.startswith('think_'):
        # think样本
        path = think_path
        network_idx = network_id[6:]  # 去除'think_'前缀（think_的长度是6）
        label = 1
    elif network_id.startswith('nothink_'):
        # nothink样本
        path = nothink_path
        network_idx = network_id[8:]  # 去除'nothink_'前缀（nothink_的长度是8）
        label = 0
    else:
        # 如果没有前缀，默认为think样本
        path = think_path
        network_idx = network_id
        label = 1
    
    print(f"直接加载样本: {path}/{network_idx}, 标签: {label}")
    # 使用现有的wrap_binary_data函数处理单个样本
    data = wrap_binary_data(path, network_idx, llm_layer, label, network_density, from_sparse_data)
    return data

# 添加模型包装类，用于GNNExplainer
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.embedding = model.embedding  # 引用原始模型的嵌入层

    def forward(self, emb_x, edge_index, edge_weight=None, batch=None):
        # 直接使用已经嵌入的特征，跳过嵌入步骤
        # 这里的emb_x已经是嵌入后的特征，不是节点索引
        if isinstance(self.model, GCNClassifier):
            # 使用GCNClassifier的前向传播逻辑
            for conv in self.model.convs:
                emb_x = conv(emb_x, edge_index, edge_weight)
                emb_x = torch.nn.functional.relu(emb_x)
                emb_x = torch.nn.functional.dropout(emb_x, p=self.model.dropout, training=self.training)
            
            mean_x = torch.zeros_like(emb_x[0]).unsqueeze(0)
            max_x = torch.zeros_like(emb_x[0]).unsqueeze(0)
            
            if batch is not None:
                # 如果提供了batch信息，使用它
                from torch_geometric.nn import global_mean_pool, global_max_pool
                mean_x = global_mean_pool(emb_x, batch)
                max_x = global_max_pool(emb_x, batch)
            else:
                # 否则，假设只有一个图，手动计算
                mean_x = emb_x.mean(dim=0, keepdim=True)
                max_x = emb_x.max(dim=0, keepdim=True)[0]
            
            x = torch.cat([mean_x, max_x], dim=1)
            x = torch.nn.functional.relu(self.model.fc1(x))
            x = self.model.fc2(x)
            return x
        
        elif isinstance(self.model, GCNClassifierLinear):
            # 使用GCNClassifierLinear的前向传播逻辑
            for conv in self.model.convs:
                emb_x = conv(emb_x, edge_index, edge_weight)
                emb_x = torch.nn.functional.dropout(emb_x, p=self.model.dropout, training=self.training)
            
            mean_x = torch.zeros_like(emb_x[0]).unsqueeze(0)
            max_x = torch.zeros_like(emb_x[0]).unsqueeze(0)
            
            if batch is not None:
                # 如果提供了batch信息，使用它
                from torch_geometric.nn import global_mean_pool, global_max_pool
                mean_x = global_mean_pool(emb_x, batch)
                max_x = global_max_pool(emb_x, batch)
            else:
                # 否则，假设只有一个图，手动计算
                mean_x = emb_x.mean(dim=0, keepdim=True)
                max_x = emb_x.max(dim=0, keepdim=True)[0]
            
            x = torch.cat([mean_x, max_x], dim=1)
            x = self.model.fc2(self.model.fc1(x))
            return x
        
        else:
            raise ValueError(f"不支持的模型类型: {type(self.model)}")

def get_explainer(model, args):
    """获取合适的解释器"""
    if args.explanation_method == 'gnnexplainer':
        algorithm = GNNExplainer(epochs=200)
    elif args.explanation_method == 'captum_ig':
        algorithm = CaptumExplainer('IntegratedGradients')
    elif args.explanation_method == 'captum_saliency':
        algorithm = CaptumExplainer('Saliency')
    else:
        raise ValueError(f"不支持的解释方法: {args.explanation_method}")
    
    explainer = Explainer(
        model=model,
        algorithm=algorithm,
        explanation_type='phenomenon' if args.explanation_type == 'phenomenon' else 'model',
        node_mask_type='object',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs' if not args.linear_probing else 'raw',
        ),
    )
    return explainer

def parse_args():
    parser = argparse.ArgumentParser(description='解释GNN模型')
    parser.add_argument('--model_path', type=str, default='saves/binary_classification/layer_14/best_model_density-1.0_dim-32_hop-1.pth',
                        help='训练好的模型路径')
    parser.add_argument('--think_path', type=str, 
                        default='data/graph_probing/data4/huguangyi/models/Qwen/Qwen3-0.6B_no_thinking',
                        help='think数据集路径')
    parser.add_argument('--nothink_path', type=str,
                        default='data/graph_probing/data4/huguangyi/models/Qwen/Qwen3-0.6B_nothink',
                        help='nothink数据集路径')
    parser.add_argument('--network_density', type=float, default=1.0,
                        help='网络密度')
    parser.add_argument('--llm_layer', type=int, default=14,
                        help='LLM的层数')
    parser.add_argument('--linear_probing', action='store_true',
                        help='是否使用线性探针')
    parser.add_argument('--num_channels', type=int, default=32,
                        help='GNN通道数')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='GNN层数')
    parser.add_argument('--sample_id', type=str, default='',
                        help='要解释的特定样本ID，格式: [think_/nothink_]<id>。如果提供，将直接加载此样本')
    parser.add_argument('--sample_idx', type=int, default=20,
                        help='从数据集加载的样本索引，仅当sample_id为空时使用')
    parser.add_argument('--load_nothink', action='store_true',
                        help='在使用sample_idx时，指定加载nothink样本而非think样本')
    parser.add_argument('--fast_mode', action='store_true',
                        help='启用快速模式，使用预先存储的样本列表避免加载整个数据集')
    parser.add_argument('--gpu_id', type=int, default=6,
                        help='GPU ID')
    parser.add_argument('--explanation_method', type=str, default='gnnexplainer',
                        choices=['gnnexplainer', 'captum_ig', 'captum_saliency'],
                        help='解释方法')
    parser.add_argument('--explanation_type', type=str, default='model',
                        choices=['model', 'phenomenon'],
                        help='解释类型: model-解释模型决策, phenomenon-解释样本标签关联的拓扑特征')
    parser.add_argument('--output_dir', type=str, default='explanation_results_nothink',
                        help='解释结果保存目录')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    # 加载数据
    if args.sample_id:
        # 如果提供了特定样本ID，直接加载单个样本
        print(f"直接加载样本 {args.sample_id}...")
        data = load_single_sample(
            args.think_path, 
            args.nothink_path, 
            args.sample_id, 
            args.llm_layer, 
            args.network_density
        ).to(device)
    elif args.fast_mode:
        # 快速模式：使用预先存储的样本列表
        print("使用快速模式加载样本列表...")
        
        # 检查样本列表缓存是否存在
        sample_list_path = f"sample_list_layer_{args.llm_layer}.pt"
        if os.path.exists(sample_list_path):
            # 加载缓存的样本列表
            sample_list = torch.load(sample_list_path)
            print(f"从缓存加载了 {len(sample_list)} 个样本索引")
            
            # 统计样本类型分布
            think_count = sum(1 for item in sample_list if item['label'] == 1)
            nothink_count = sum(1 for item in sample_list if item['label'] == 0)
            print(f"样本分布: think样本数量: {think_count}, nothink样本数量: {nothink_count}")
            print(f"样本0-10的标签: {[sample_list[i]['label'] for i in range(min(10, len(sample_list)))]}")
            print(f"样本40-60的标签: {[sample_list[i]['label'] for i in range(40, min(60, len(sample_list)))]}")
            
            # 获取请求的样本
            sample_idx = min(args.sample_idx, len(sample_list) - 1)
            sample_info = sample_list[sample_idx]
            print(f"选择的样本索引: {sample_idx}, ID: {sample_info['id']}, 标签: {sample_info['label']}")
            
            # 如果用户指定要加载nothink样本，但当前样本是think样本，或者相反
            if (args.load_nothink and sample_info['label'] == 1) or (not args.load_nothink and sample_info['label'] == 0):
                # 查找对应类别的样本
                target_label = 0 if args.load_nothink else 1
                matching_samples = [i for i, item in enumerate(sample_list) if item['label'] == target_label]
                
                if matching_samples:
                    # 选择索引最接近的样本
                    closest_idx = min(matching_samples, key=lambda x: abs(x - sample_idx))
                    sample_info = sample_list[closest_idx]
                    sample_idx = closest_idx  # 更新sample_idx为新选择的索引
                    print(f"根据--load_nothink参数调整: 选择的新样本索引: {closest_idx}, ID: {sample_info['id']}, 标签: {sample_info['label']}")
                else:
                    print(f"警告: 无法找到目标标签为{target_label}的样本，仍使用原样本")
            
            # 加载单个样本
            data = load_single_sample(
                args.think_path,
                args.nothink_path,
                sample_info['id'],
                args.llm_layer,
                args.network_density
            ).to(device)
        else:
            # 如果没有缓存，需要加载完整数据集并创建缓存
            print("未找到样本列表缓存，将加载完整数据集并创建缓存...")
            _, test_data_loader = get_binary_classification_dataloader(
                think_path=args.think_path,
                nothink_path=args.nothink_path,
                network_density=args.network_density,
                from_sparse_data=False,
                llm_layer=args.llm_layer,
                batch_size=1,
                eval_batch_size=1,
                num_workers=1,
                prefetch_factor=1,
                test_set_ratio=0.2,
                in_memory=True,
            )
            
            # 创建样本列表缓存
            sample_list = []
            for i, sample_data in enumerate(test_data_loader.dataset):
                is_think = sample_data.y.item() == 1
                prefix = "think_" if is_think else "nothink_"
                sample_list.append({
                    'idx': i,
                    'id': f"{prefix}{i}",  # 创建唯一ID
                    'label': sample_data.y.item()
                })
            
            # 保存缓存
            torch.save(sample_list, sample_list_path)
            print(f"创建并保存了包含 {len(sample_list)} 个样本的列表")
            
            # 获取请求的样本
            sample_idx = min(args.sample_idx, len(sample_list) - 1)
            sample_info = sample_list[sample_idx]
            print(f"选择的样本索引: {sample_idx}, ID: {sample_info['id']}, 标签: {sample_info['label']}")
            
            # 如果用户指定要加载nothink样本，但当前样本是think样本，或者相反
            if (args.load_nothink and sample_info['label'] == 1) or (not args.load_nothink and sample_info['label'] == 0):
                # 查找对应类别的样本
                target_label = 0 if args.load_nothink else 1
                matching_samples = [i for i, item in enumerate(sample_list) if item['label'] == target_label]
                
                if matching_samples:
                    # 选择索引最接近的样本
                    closest_idx = min(matching_samples, key=lambda x: abs(x - sample_idx))
                    sample_info = sample_list[closest_idx]
                    sample_idx = closest_idx  # 更新sample_idx为新选择的索引
                    print(f"根据--load_nothink参数调整: 选择的新样本索引: {closest_idx}, ID: {sample_info['id']}, 标签: {sample_info['label']}")
                else:
                    print(f"警告: 无法找到目标标签为{target_label}的样本，仍使用原样本")
            
            # 加载单个样本
            data = test_data_loader.dataset[sample_idx].to(device)
    else:
        # 常规模式：加载完整数据集
        print(f"加载完整数据集...")
        _, test_data_loader = get_binary_classification_dataloader(
            think_path=args.think_path,
            nothink_path=args.nothink_path,
            network_density=args.network_density,
            from_sparse_data=False,
            llm_layer=args.llm_layer,
            batch_size=1,
            eval_batch_size=1,
            num_workers=1,
            prefetch_factor=1,
            test_set_ratio=0.2,
            in_memory=True,
        )
        
        # 获取单个样本
        print(f"获取样本数据...")
        sample_idx = min(args.sample_idx, len(test_data_loader.dataset) - 1)
        data = test_data_loader.dataset[sample_idx].to(device)
        
        # 检查是否需要根据--load_nothink参数调整
        is_think = data.y.item() == 1
        if (args.load_nothink and is_think) or (not args.load_nothink and not is_think):
            # 需要切换样本类型
            target_label = 0 if args.load_nothink else 1
            print(f"当前样本标签为 {data.y.item()}, 根据--load_nothink参数需要查找标签为 {target_label} 的样本")
            
            # 查找对应标签的样本
            matching_samples = []
            for i, sample_data in enumerate(test_data_loader.dataset):
                if sample_data.y.item() == target_label:
                    matching_samples.append(i)
            
            if matching_samples:
                # 选择索引最接近的样本
                closest_idx = min(matching_samples, key=lambda x: abs(x - sample_idx))
                print(f"找到匹配的样本，索引: {closest_idx}")
                data = test_data_loader.dataset[closest_idx].to(device)
            else:
                print(f"警告: 无法找到标签为 {target_label} 的样本，仍使用原样本")
    
    # 初始化模型
    print(f"初始化模型...")
    if not args.linear_probing:
        model = GCNClassifier(
            num_nodes=llm_model_num_nodes_map.get("qwen3-0.6b", 1024),
            hidden_channels=args.num_channels,
            out_channels=args.num_channels,
            num_layers=args.num_layers,
            dropout=0.0,
        ).to(device)
    else:
        model = GCNClassifierLinear(
            num_nodes=llm_model_num_nodes_map.get("qwen3-0.6b", 1024),
            hidden_channels=args.num_channels,
            out_channels=args.num_channels,
            num_layers=args.num_layers,
            dropout=0.0,
        ).to(device)
    
    # 加载模型权重
    print(f"加载模型权重: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # 数据检查
    print(f"数据检查：x形状: {data.x.shape}, edge_index形状: {data.edge_index.shape}")
    print(f"样本标签: {data.y.item()} (0:nothink, 1:think)")
    
    # 预先应用嵌入
    with torch.no_grad():
        # 应用嵌入层处理，获取嵌入后的特征
        embedded_x = model.embedding(data.x)
        print(f"嵌入后的特征形状: {embedded_x.shape}")
    
    # 创建包装模型
    wrapped_model = ModelWrapper(model).to(device)
    wrapped_model.eval()
    
    # 创建解释器
    print(f"创建{args.explanation_method}解释器 (类型: {args.explanation_type})...")
    explainer = get_explainer(wrapped_model, args)
    
    # 生成解释
    print(f"生成解释...")
    if args.explanation_type == 'phenomenon':
        # 对于phenomenon类型，将样本自己的标签作为target参数
        sample_label = data.y
        print(f"使用样本自身的标签 {sample_label.item()} (0:nothink, 1:think) 进行现象解释")
        
        # 确保target是正确的形状和类型(LongTensor)
        if sample_label.dim() == 0:  # 如果是标量
            sample_label = sample_label.unsqueeze(0)
        
        # 确保类型是Long，这对于nll_loss是必要的
        sample_label = sample_label.long()
        print(f"Target形状: {sample_label.shape}, 类型: {sample_label.dtype}")
            
        explanation = explainer(
            embedded_x,  # 使用预先嵌入的特征
            data.edge_index, 
            edge_weight=data.edge_attr,
            target=sample_label,  # 使用样本的实际标签作为target
            batch=torch.zeros(data.x.size(0), dtype=torch.long, device=device)
        )
    else:
        # 对于model类型，不需要target参数
        explanation = explainer(
            embedded_x,  # 使用预先嵌入的特征
            data.edge_index, 
            edge_weight=data.edge_attr,
            batch=torch.zeros(data.x.size(0), dtype=torch.long, device=device)
        )
    
    # 打印解释结果
    print(f"\n解释结果摘要:")
    print(f"边掩码形状: {explanation.edge_mask.shape}")
    print(f"节点掩码形状: {explanation.node_mask.shape}")
    
    # 节点重要性前10名
    node_importance = explanation.node_mask.sum(dim=1)
    top_nodes_idx = torch.argsort(node_importance, descending=True)[:10]
    print(f"\n节点重要性排名前10:")
    for i, idx in enumerate(top_nodes_idx):
        print(f"排名 {i+1}: 节点 {idx.item()}, 重要性: {node_importance[idx].item():.4f}")
    
    # 边重要性前10名
    edge_importance = explanation.edge_mask
    top_edges_idx = torch.argsort(edge_importance, descending=True)[:10]
    print(f"\n边重要性排名前10:")
    for i, idx in enumerate(top_edges_idx):
        src = data.edge_index[0, idx].item()
        dst = data.edge_index[1, idx].item()
        print(f"排名 {i+1}: 边 ({src}, {dst}), 重要性: {edge_importance[idx].item():.4f}")
    
    # 保存可视化结果
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存边掩码和节点掩码
    torch.save(explanation.edge_mask, os.path.join(args.output_dir, f"edge_mask_{args.explanation_method}.pt"))
    torch.save(explanation.node_mask, os.path.join(args.output_dir, f"node_mask_{args.explanation_method}.pt"))
    
    # 子图可视化
    try:
        G = to_networkx(data, to_undirected=True)
        pos = nx.spring_layout(G, seed=42)  # 添加随机种子以确保布局一致性
        
        # 修复边权重归一化
        edge_weights = explanation.edge_mask.cpu().numpy()
        # 确保边权重在[0,1]范围内，避免RGBA参数错误
        edge_weights_norm = np.clip((edge_weights - edge_weights.min()) / 
                               (edge_weights.max() - edge_weights.min() + 1e-8), 0, 1)
        
        plt.figure(figsize=(12, 10))
        
        # 绘制节点，根据节点重要性调整大小
        node_importance = explanation.node_mask.sum(dim=1).cpu().numpy()
        node_importance_norm = (node_importance - node_importance.min()) / (node_importance.max() - node_importance.min() + 1e-8)
        node_size = [50 + 200 * imp for imp in node_importance_norm]
        
        # 突出显示前10个重要节点
        top_nodes_idx = torch.argsort(explanation.node_mask.sum(dim=1), descending=True)[:10].cpu().numpy()
        node_colors = ['red' if i in top_nodes_idx else 'skyblue' for i in range(len(G.nodes()))]
        
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_colors, alpha=0.8)
        
        # 为了清晰起见，只绘制重要性较高的边
        important_edges_idx = np.where(edge_weights_norm > np.percentile(edge_weights_norm, 95))[0]
        important_edges = [(data.edge_index[0, i].item(), data.edge_index[1, i].item()) for i in important_edges_idx]
        important_weights = edge_weights_norm[important_edges_idx]
        
        # 绘制重要边
        if len(important_edges) > 0:
            nx.draw_networkx_edges(
                G, pos, 
                edgelist=important_edges,
                width=[1 + 5 * w for w in important_weights],
                edge_color=important_weights,
                edge_cmap=plt.cm.Blues,
                alpha=0.6
            )
        
        # 添加节点标签（仅为前10个重要节点）
        labels = {node: str(node) for node in top_nodes_idx}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=12, font_weight='bold')
        
        plt.title('GNN Explanation: Most Important Nodes and Edges')
        plt.axis('off')
        plt.tight_layout()
        
        # 注意：可能会出现NumPy弃用警告 'alltrue is deprecated' - 这是NetworkX库内部使用的方法，
        # 将在未来版本中更新。这个警告不影响可视化结果。
        output_path = os.path.join(args.output_dir, f"graph_explanation_{args.explanation_method}.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=300, format='png')
        plt.close()
        print(f"图解释可视化已保存到 {output_path}")
    except Exception as e:
        print(f"图可视化失败: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n解释完成，结果已保存到 {args.output_dir} 目录")

if __name__ == "__main__":
    main() 