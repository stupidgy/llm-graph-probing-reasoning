import os
import pickle
from tqdm import tqdm

import numpy as np

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse

def wrap_binary_data(path, network_id, llm_layer, label, network_density, from_sparse_data=False):
    """
    封装单个图数据，设置二分类标签（0或1）
    """
    if not from_sparse_data:
        network = np.load(os.path.join(path, f"{network_id}/layer_{llm_layer}_corr.npy")).astype(np.float32)
        percentile_threshold = network_density * 100
        threshold = np.percentile(np.abs(network), 100 - percentile_threshold)
        network[np.abs(network) < threshold] = 0
        np.fill_diagonal(network, 1.0)
        llm_brain_network = torch.from_numpy(network)
        edge_index_llm, edge_attr_llm = dense_to_sparse(llm_brain_network)
        num_nodes = llm_brain_network.shape[0]
    else:
        edge_index_llm = np.load(os.path.join(path, f"{network_id}/layer_{llm_layer}_sparse_{network_density}_edge_index.npy"))
        edge_attr_llm = np.load(os.path.join(path, f"{network_id}/layer_{llm_layer}_sparse_{network_density}_edge_attr.npy")).astype(np.float32)
        num_nodes = edge_index_llm.max() + 1
        edge_index_llm = torch.from_numpy(edge_index_llm)
        edge_attr_llm = torch.from_numpy(edge_attr_llm)
    
    # 使用二分类标签（0或1）
    data = Data(
        x=torch.arange(num_nodes),
        edge_index=edge_index_llm,
        edge_attr=edge_attr_llm,
        y=torch.tensor([label], dtype=torch.float32)
    )
    return data


class BinaryClassificationDataset(Dataset):
    """
    用于二分类任务的数据集类
    """
    def __init__(self, network_ids, dataset_paths, labels, llm_layer, network_density, from_sparse_data=False):
        super().__init__(None, transform=None, pre_transform=None)
        self.network_ids = network_ids  # 网络ID列表
        self.dataset_paths = dataset_paths  # 对应的数据路径列表
        self.labels = labels  # 标签列表：0表示nothink，1表示think
        self.llm_layer = llm_layer
        self.network_density = network_density
        self.from_sparse_data = from_sparse_data

    def len(self):
        return len(self.network_ids)

    def get(self, idx):
        network_id = self.network_ids[idx]
        path = self.dataset_paths[idx]
        label = self.labels[idx]
        data = wrap_binary_data(path, network_id, self.llm_layer, label, self.network_density, self.from_sparse_data)
        return data


def get_binary_classification_dataloader(
    think_path="data/graph_probing/data4/huguangyi/models/Qwen/Qwen3-0.6B",
    nothink_path="data/graph_probing/data4/huguangyi/models/Qwen/Qwen3-0.6B_nothink",
    network_density=1.0,
    from_sparse_data=False,
    llm_layer=0,
    batch_size=32,
    eval_batch_size=32,
    num_workers=4,
    prefetch_factor=2,
    test_set_ratio=0.2,
    in_memory=True,
    shuffle=True,
    **kwargs
):
    """
    创建二分类任务的数据加载器
    从think和nothink目录中读取数据，分配标签（think为1，nothink为0）
    
    参数:
        think_path: think数据集路径，标签为1
        nothink_path: nothink数据集路径，标签为0
        network_density: 网络密度
        from_sparse_data: 是否从稀疏数据读取
        llm_layer: 模型层数
        batch_size: 训练批次大小
        eval_batch_size: 评估批次大小
        num_workers: 数据加载线程数
        prefetch_factor: 预取因子
        test_set_ratio: 测试集比例
        in_memory: 是否在内存中加载全部数据
        shuffle: 是否打乱数据
        
    返回:
        train_data_loader: 训练数据加载器
        test_data_loader: 测试数据加载器
    """
    # 获取think目录中的所有子目录（即网络ID）
    think_ids = sorted([d for d in os.listdir(think_path) if os.path.isdir(os.path.join(think_path, d))])
    
    # 获取nothink目录中的所有子目录
    nothink_ids = sorted([d for d in os.listdir(nothink_path) if os.path.isdir(os.path.join(nothink_path, d))])
    
    # 创建数据ID、路径和标签列表
    network_ids = []
    dataset_paths = []
    labels = []
    
    # 添加think数据（标签为1）
    for network_id in think_ids:
        network_ids.append(network_id)
        dataset_paths.append(think_path)
        labels.append(1)  # think标签为1
    
    # 添加nothink数据（标签为0）
    for network_id in nothink_ids:
        network_ids.append(network_id)
        dataset_paths.append(nothink_path)
        labels.append(0)  # nothink标签为0
    
    # 随机分割训练集和测试集
    indices = list(range(len(network_ids)))
    generator = torch.Generator().manual_seed(42)
    test_set_size = int(len(indices) * test_set_ratio)
    train_indices, test_indices = torch.utils.data.random_split(
        indices, [len(indices) - test_set_size, test_set_size], generator=generator)
    
    if in_memory:
        data_list = []
        for i, (network_id, path, label) in enumerate(zip(network_ids, dataset_paths, labels)):
            try:
                data = wrap_binary_data(path, network_id, llm_layer, label, network_density, from_sparse_data)
                data_list.append(data)
            except Exception as e:
                print(f"无法加载数据 {network_id} 从 {path}: {e}")
        
        train_dataset = [data_list[i] for i in train_indices]
        test_dataset = [data_list[i] for i in test_indices]
    else:
        # 创建训练集
        train_network_ids = [network_ids[i] for i in train_indices]
        train_dataset_paths = [dataset_paths[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        
        train_dataset = BinaryClassificationDataset(
            train_network_ids,
            train_dataset_paths,
            train_labels,
            llm_layer,
            network_density,
            from_sparse_data
        )
        
        # 创建测试集
        test_network_ids = [network_ids[i] for i in test_indices]
        test_dataset_paths = [dataset_paths[i] for i in test_indices]
        test_labels = [labels[i] for i in test_indices]
        
        test_dataset = BinaryClassificationDataset(
            test_network_ids,
            test_dataset_paths,
            test_labels,
            llm_layer,
            network_density,
            from_sparse_data
        )
    
    # 创建数据加载器
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        **kwargs
    )
    
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        **kwargs
    )
    
    return train_data_loader, test_data_loader
