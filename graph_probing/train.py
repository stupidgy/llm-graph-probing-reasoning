from absl import app, flags, logging
import os
from setproctitle import setproctitle
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
torch.set_default_dtype(torch.float32)
from torch.utils.tensorboard import SummaryWriter
from torch.cuda import amp  # 导入amp模块用于混合精度训练

from graph_probing.dataset import get_binary_classification_dataloader
from graph_probing.model import GCNClassifier, GCNClassifierLinear
from graph_probing.utils import llm_model_num_nodes_map, test_classification_fn

flags.DEFINE_string("think_path", "data/graph_probing/data4/huguangyi/models/Qwen/Qwen3-0.6B", "The path to the think dataset.")
flags.DEFINE_string("nothink_path", "data/graph_probing/data4/huguangyi/models/Qwen/Qwen3-0.6B_nothink", "The path to the nothink dataset.")
flags.DEFINE_float("network_density", 1.0, "The density of the network.")
flags.DEFINE_boolean("from_sparse_data", False, "Whether to use sparse data.")
flags.DEFINE_integer("llm_layer", 21, "The layer of the LLM model.")
flags.DEFINE_integer("batch_size", 1, "The batch size.")
flags.DEFINE_integer("eval_batch_size", 1, "The evaluation batch size.")
flags.DEFINE_integer("num_workers", 4, "Number of workers.")
flags.DEFINE_integer("prefatch_factor", 4, "Prefetch factor.")
flags.DEFINE_boolean("linear_probing", True, "Whether to use linear probing.")
flags.DEFINE_integer("num_channels", 32, "The number of channels in GNN probes.")
flags.DEFINE_integer("num_layers", 1, "The number of GNN layers.")
flags.DEFINE_float("dropout", 0.0, "The dropout rate.")
flags.DEFINE_float("lr", 0.001, "The learning rate.")
flags.DEFINE_float("weight_decay", 1e-5, "The weight decay.")
flags.DEFINE_integer("num_epochs", 100, "The number of epochs.")
flags.DEFINE_float("test_set_ratio", 0.2, "The ratio of the test set.")
flags.DEFINE_boolean("in_memory", True, "In-memory dataset.")
flags.DEFINE_integer("early_stop_patience", 20, "The patience for early stopping.")
flags.DEFINE_integer("gpu_id", 6, "The GPU ID.")
flags.DEFINE_boolean("use_fp16", False, "使用FP16混合精度训练")
flags.DEFINE_boolean("resume", False, "Whether to resume training from the best model.")
FLAGS = flags.FLAGS


def train_classification_model(model, train_data_loader, test_data_loader, optimizer, scheduler, writer, save_model_path, device):
    # 设置混合精度训练
    scaler = amp.GradScaler(enabled=FLAGS.use_fp16 and device.type == 'cuda')
    
    # 初始评估
    accuracy, precision, recall, f1, confusion_mat = test_classification_fn(model, test_data_loader, device, use_fp16=FLAGS.use_fp16)
    torch.cuda.empty_cache()
    
    logging.info(f"Initial Test Accuracy: {accuracy:.4f}")
    logging.info(f"Initial Test Precision: {precision:.4f}")
    logging.info(f"Initial Test Recall: {recall:.4f}")
    logging.info(f"Initial Test F1 Score: {f1:.4f}")
    logging.info(f"Initial Confusion Matrix:\n{confusion_mat}")
    
    writer.add_scalar("test/accuracy", accuracy, 0)
    writer.add_scalar("test/precision", precision, 0)
    writer.add_scalar("test/recall", recall, 0)
    writer.add_scalar("test/f1", f1, 0)

    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)

    # 保存最佳模型的指标
    best_accuracy = accuracy
    best_f1 = f1
    best_epoch = 0
    epochs_no_improve = 0
    
    # 训练循环
    for epoch in tqdm(range(FLAGS.num_epochs), position=0, desc="Training"):
        model.train()
        total_loss = 0.0
        num_graphs = 0
        correct = 0
        
        for data in tqdm(train_data_loader, position=1, desc=f"Epoch {epoch + 1}", leave=False):
            optimizer.zero_grad()
            
            # 将数据移动到设备
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            edge_attr = data.edge_attr.to(device)
            batch = data.batch.to(device)
            target = data.y.to(device).long()
            
            # 使用混合精度训练
            with amp.autocast(enabled=FLAGS.use_fp16 and device.type == 'cuda'):
                logits = model(x, edge_index, edge_attr, batch)
                loss = F.cross_entropy(logits, target)
            
            # 使用scaler处理反向传播和优化
            if FLAGS.use_fp16 and device.type == 'cuda':
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            # 统计准确率
            pred = logits.argmax(dim=1)
            correct += (pred == target).sum().item()
            
            num_graphs += data.num_graphs
            total_loss += loss.item() * data.num_graphs
            
        avg_loss = total_loss / num_graphs
        train_accuracy = correct / num_graphs
        
        logging.info(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.4f}")
        writer.add_scalar("train/loss", avg_loss, epoch + 1)
        writer.add_scalar("train/accuracy", train_accuracy, epoch + 1)
        torch.cuda.empty_cache()

        # 测试评估
        accuracy, precision, recall, f1, confusion_mat = test_classification_fn(model, test_data_loader, device, use_fp16=FLAGS.use_fp16)
        torch.cuda.empty_cache()
        
        logging.info(f"Test Accuracy: {accuracy:.4f}")
        logging.info(f"Test Precision: {precision:.4f}")
        logging.info(f"Test Recall: {recall:.4f}")
        logging.info(f"Test F1 Score: {f1:.4f}")
        logging.info(f"Confusion Matrix:\n{confusion_mat}")
        
        writer.add_scalar("test/accuracy", accuracy, epoch + 1)
        writer.add_scalar("test/precision", precision, epoch + 1)
        writer.add_scalar("test/recall", recall, epoch + 1)
        writer.add_scalar("test/f1", f1, epoch + 1)
        
        # 使用F1分数作为学习率调度器的指标
        scheduler.step(1.0 - f1)  # 最小化1-f1，相当于最大化f1

        # 保存最佳模型
        if f1 > best_f1 or (f1 == best_f1 and accuracy > best_accuracy):
            best_accuracy = accuracy
            best_f1 = f1
            best_epoch = epoch + 1
            torch.save(model.state_dict(), save_model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= FLAGS.early_stop_patience:
                logging.info(f"Early stopping at epoch {epoch + 1}")
                break

    logging.info(f"Best Epoch: {best_epoch}")
    logging.info(f"Best Test Accuracy: {best_accuracy:.4f}")
    logging.info(f"Best Test F1 Score: {best_f1:.4f}")

    writer.add_text(
        "best_record",
        f"Best Epoch: {best_epoch}, "
        f"Best Test Accuracy: {best_accuracy:.4f}, "
        f"Best Test F1 Score: {best_f1:.4f}",
        0
    )


def main(_):
    device = torch.device(f"cuda:{FLAGS.gpu_id}")

    # 如果使用FP16，记录
    if FLAGS.use_fp16:
        logging.info("使用FP16混合精度训练")
    
    # 加载二分类数据
    train_data_loader, test_data_loader = get_binary_classification_dataloader(
        think_path=FLAGS.think_path,
        nothink_path=FLAGS.nothink_path,
        network_density=FLAGS.network_density,
        from_sparse_data=FLAGS.from_sparse_data,
        llm_layer=FLAGS.llm_layer,
        batch_size=FLAGS.batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        num_workers=FLAGS.num_workers,
        prefetch_factor=FLAGS.prefatch_factor,
        test_set_ratio=FLAGS.test_set_ratio,
        in_memory=FLAGS.in_memory,
    )

    # 初始化分类模型
    if not FLAGS.linear_probing:
        model = GCNClassifier(
            num_nodes=llm_model_num_nodes_map.get("qwen3-0.6b", 1024),  # 默认使用qwen2.5-0.5b的节点数
            hidden_channels=FLAGS.num_channels,
            out_channels=FLAGS.num_channels,
            num_layers=FLAGS.num_layers,
            dropout=FLAGS.dropout,
        ).to(device)
        model_name = "binary_classification"   # 图卷积神经网络
        save_dir = f"saves/{model_name}/layer_{FLAGS.llm_layer}"
    else:
        model = GCNClassifierLinear(
            num_nodes=llm_model_num_nodes_map.get("qwen3-0.6b", 1024),
            hidden_channels=FLAGS.num_channels,
            out_channels=FLAGS.num_channels,
            num_layers=FLAGS.num_layers,
            dropout=FLAGS.dropout,
        ).to(device)
        model_name = "binary_classification_linear"   # 线性探针
        save_dir = f"saves/{model_name}/layer_{FLAGS.llm_layer}"

    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6)
    writer = SummaryWriter(log_dir=f"runs/{model_name}/layer_{FLAGS.llm_layer}")
    
    writer.add_hparams(
        {
            "linear_probing": FLAGS.linear_probing,
            "hidden_channels": FLAGS.num_channels, "out_channels": FLAGS.num_channels,
            "num_layers": FLAGS.num_layers, "dropout": FLAGS.dropout,
            "batch_size": FLAGS.batch_size, "lr": FLAGS.lr, "weight_decay": FLAGS.weight_decay,
            "use_fp16": FLAGS.use_fp16
        },
        {"hparam/placeholder": 0}
    )

    # 模型保存路径
    save_model_path = os.path.join(
        save_dir,
        f"best_model_density-{FLAGS.network_density}_dim-{FLAGS.num_channels}_hop-{FLAGS.num_layers}.pth"
    )

    # 如果需要从现有模型恢复训练
    if FLAGS.resume and os.path.exists(save_model_path):
        model.load_state_dict(torch.load(save_model_path, map_location=device, weights_only=True))
        logging.info(f"Resumed from {save_model_path}")
    
    # 训练模型
    train_classification_model(model, train_data_loader, test_data_loader, optimizer, scheduler, writer, save_model_path, device)


if __name__ == "__main__":
    setproctitle("think-nothink-classification")
    app.run(main)
