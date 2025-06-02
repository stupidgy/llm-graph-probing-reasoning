from absl import app, flags
import os
from setproctitle import setproctitle

import numpy as np
import torch
torch.set_default_dtype(torch.float32)

from graph_probing.dataset import get_binary_classification_dataloader
from graph_probing.model import GCNClassifier, GCNClassifierLinear
from graph_probing.utils import llm_model_num_nodes_map, eval_classification_model


flags.DEFINE_string("think_path", "data/graph_probing/data4/huguangyi/models/Qwen/Qwen3-0.6B", "The path to the think dataset.")
flags.DEFINE_string("nothink_path", "data/graph_probing/data4/huguangyi/models/Qwen/Qwen3-0.6B_nothink", "The path to the nothink dataset.")
flags.DEFINE_float("network_density", 1.0, "The density of the network.")
flags.DEFINE_boolean("from_sparse_data", False, "Whether to use sparse data.")
flags.DEFINE_integer("llm_layer", 0, "The layer of the LLM model.")
flags.DEFINE_integer("batch_size", 16, "The batch size.")
flags.DEFINE_integer("eval_batch_size", 16, "The evaluation batch size.")
flags.DEFINE_integer("num_workers", 4, "Number of workers.")
flags.DEFINE_integer("prefetch_factor", 4, "Prefetch factor.")
flags.DEFINE_boolean("linear_probing", False, "Whether to use linear probing.")
flags.DEFINE_integer("num_channels", 32, "The number of channels in GNN probes.")
flags.DEFINE_integer("num_layers", 1, "The number of GNN layers.")
flags.DEFINE_float("dropout", 0.0, "The dropout rate.")
flags.DEFINE_float("test_set_ratio", 0.2, "The size of the test set.")
flags.DEFINE_boolean("in_memory", True, "In-memory dataset.")
flags.DEFINE_integer("gpu_id", 7, "The GPU ID.")
FLAGS = flags.FLAGS


def main(_):

    device = torch.device(f"cuda:{FLAGS.gpu_id}")

    # 加载测试数据
    _, test_data_loader = get_binary_classification_dataloader(
        think_path=FLAGS.think_path,
        nothink_path=FLAGS.nothink_path,
        network_density=FLAGS.network_density,
        from_sparse_data=FLAGS.from_sparse_data,
        llm_layer=FLAGS.llm_layer,
        batch_size=FLAGS.batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        num_workers=FLAGS.num_workers,
        prefetch_factor=FLAGS.prefetch_factor,
        test_set_ratio=FLAGS.test_set_ratio,
        in_memory=FLAGS.in_memory,
    )

    # 初始化分类模型
    if not FLAGS.linear_probing:
        model = GCNClassifier(
            num_nodes=llm_model_num_nodes_map.get("qwen2.5-0.5b", 896),  # 默认使用qwen2.5-0.5b的节点数
            hidden_channels=FLAGS.num_channels,
            out_channels=FLAGS.num_channels,
            num_layers=FLAGS.num_layers,
            dropout=FLAGS.dropout,
        ).to(device)
    else:
        model = GCNClassifierLinear(
            num_nodes=llm_model_num_nodes_map.get("qwen2.5-0.5b", 896),
            hidden_channels=FLAGS.num_channels,
            out_channels=FLAGS.num_channels,
            num_layers=FLAGS.num_layers,
            dropout=FLAGS.dropout,
        ).to(device)
    
    # 设置模型保存和结果保存路径
    model_name = "binary_classification"
    save_dir = f"saves/{model_name}/layer_{FLAGS.llm_layer}"
    
    model_save_path = os.path.join(
        save_dir, 
        f"best_model_density-{FLAGS.network_density}_dim-{FLAGS.num_channels}_hop-{FLAGS.num_layers}.pth"
    )
    
    # 加载训练好的模型权重
    model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))

    # 评估模型并获取结果
    all_y, all_pred = eval_classification_model(model, test_data_loader, device)
    
    # 保存评估结果
    os.makedirs(save_dir, exist_ok=True)
    results = np.vstack((all_y, all_pred))
    np.save(f"{save_dir}/classification_results_density-{FLAGS.network_density}_dim-{FLAGS.num_channels}_hop-{FLAGS.num_layers}.npy", results)


if __name__ == "__main__":
    setproctitle("think-nothink-classification-eval")
    app.run(main)
