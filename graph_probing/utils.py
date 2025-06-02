from absl import logging
from tqdm import tqdm
import numpy as np
from sklearn.metrics import r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
import torch.nn.functional as F
from scipy import stats


hf_model_name_map = {
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2-medium",
    "gpt2-large": "gpt2-large",
    "pythia-160m": "EleutherAI/pythia-160m",
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
    "pythia-2.8b": "EleutherAI/pythia-2.8b",
    "pythia-6.9b": "EleutherAI/pythia-6.9b",
    "pythia-12b": "EleutherAI/pythia-12b",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B",
}


llm_model_num_nodes_map = {
    "gpt2": 768,
    "gpt2-medium": 1024,
    "gpt2-large": 1280,
    "pythia-160m": 768,
    "pythia-410m": 1024,
    "pythia-1.4b": 2048,
    "pythia-2.8b": 2560,
    "pythia-6.9b": 4096,
    "pythia-12b": 5120,
    "qwen2.5-0.5b": 896,
    "qwen2.5-3b": 2048,
    "qwen2.5-7b": 3584,
    "qwen2.5-14b": 5120,
    "qwen3-0.6b": 1024,
}

# 以下是为二分类任务添加的新函数
def test_classification_fn(model, test_data_loader, device, return_raw_data=False, use_fp16=False):
    """
    用于评估分类模型的函数
    
    参数:
        model: 分类模型
        test_data_loader: 测试数据加载器
        device: 设备（CPU/GPU）
        return_raw_data: 是否返回原始预测和标签数据
        use_fp16: 是否使用FP16混合精度
        
    返回:
        accuracy: 准确率
        precision: 精确率
        recall: 召回率
        f1: F1分数
        confusion_mat: 混淆矩阵
        [可选] all_y: 所有真实标签
        [可选] all_pred_labels: 所有预测标签
    """
    model.eval()
    with torch.no_grad():
        all_pred_logits = []
        all_pred_labels = []
        all_y = []
        for data in tqdm(test_data_loader, desc="Testing Classification", leave=False):
            # 将数据移动到设备
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            edge_attr = data.edge_attr.to(device)
            batch = data.batch.to(device)
            y = data.y.to(device)
            
            # 使用混合精度
            if use_fp16 and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    logits = model(x, edge_index, edge_attr, batch)
            else:
                logits = model(x, edge_index, edge_attr, batch)
                
            pred_labels = torch.argmax(logits, dim=1)
            
            all_pred_logits.append(logits.cpu().detach())
            all_pred_labels.append(pred_labels.cpu().detach())
            all_y.append(y.cpu().detach().long())
            
        all_pred_logits = torch.cat(all_pred_logits, dim=0)
        all_pred_labels = torch.cat(all_pred_labels, dim=0).numpy()
        all_y = torch.cat(all_y, dim=0).numpy()
        
        # 计算评估指标
        accuracy = accuracy_score(all_y, all_pred_labels)
        precision = precision_score(all_y, all_pred_labels, average='binary')
        recall = recall_score(all_y, all_pred_labels, average='binary')
        f1 = f1_score(all_y, all_pred_labels, average='binary')
        confusion_mat = confusion_matrix(all_y, all_pred_labels)

    if not return_raw_data:
        return accuracy, precision, recall, f1, confusion_mat
    else:
        return accuracy, precision, recall, f1, confusion_mat, all_y, all_pred_labels


def eval_classification_model(model, test_data_loader, device):
    """
    评估分类模型并打印结果
    
    参数:
        model: 分类模型
        test_data_loader: 测试数据加载器
        device: 设备（CPU/GPU）
        
    返回:
        all_y: 所有真实标签
        all_pred_labels: 所有预测标签
    """
    accuracy, precision, recall, f1, confusion_mat, all_y, all_pred_labels = test_classification_fn(
        model, test_data_loader, device, return_raw_data=True
    )
    torch.cuda.empty_cache()
    
    logging.info(f"Test Accuracy: {accuracy:.4f}")
    logging.info(f"Test Precision: {precision:.4f}")
    logging.info(f"Test Recall: {recall:.4f}")
    logging.info(f"Test F1 Score: {f1:.4f}")
    logging.info(f"Confusion Matrix:\n{confusion_mat}")

    return all_y, all_pred_labels


