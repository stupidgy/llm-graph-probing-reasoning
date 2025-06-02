from absl import app, flags
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pickle
import json
from setproctitle import setproctitle
from tqdm import tqdm

from multiprocessing import Process, Queue
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from graph_probing.utils import hf_model_name_map

flags.DEFINE_string("dataset_filename", "data/graph_probing/openwebtext-10k-gpt2.pkl", "The dataset filename.")
flags.DEFINE_string("jsonl_filename", "/data4/huguangyi/projects/llm-graph-probing/think_common.jsonl", "The JSONL file containing the problems with thinking.")
flags.DEFINE_string("llm_model_name", "/data4/huguangyi/models/Qwen/Qwen3-0.6B", "The name of the LLM model.")
flags.DEFINE_integer("ckpt_step", -1, "The checkpoint step.")
flags.DEFINE_multi_integer("llm_layer", [0,7,21], "Layer IDs for network construction.")
flags.DEFINE_integer("batch_size", 1, "Batch size.")
flags.DEFINE_multi_integer("gpu_id", [4,5,6], "The GPU ID.")
flags.DEFINE_integer("num_workers", 10, "Number of processes for computing networks.")
flags.DEFINE_boolean("resume", False, "Resume from the last generation.")
flags.DEFINE_float("network_density", 1.0, "The density of the network.")
flags.DEFINE_boolean("use_jsonl", True, "Use JSONL file instead of pickle file.")
FLAGS = flags.FLAGS


def run_llm(
    rank,
    num_producers,
    queue,
    dataset_filename,
    model_name,
    ckpt_step,
    gpu_id,
    batch_size,
    layer_list,
    resume,
    p_save_path,
    use_jsonl=True,
    jsonl_filename="think_common.jsonl",
):
    padding_side = "right" if model_name.startswith("gpt2") else "left"
    if ckpt_step == -1:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side=padding_side)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=f"cuda:{gpu_id}", torch_dtype=torch.float16)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side=padding_side, revision=f'step{ckpt_step}')
        model = AutoModelForCausalLM.from_pretrained(model_name, revision=f'step{ckpt_step}', device_map=f"cuda:{gpu_id}", torch_dtype=torch.float16)

    if use_jsonl:
        # 从JSONL文件读取数据
        original_input_texts = []
        with open(jsonl_filename, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                # 组合问题、思考过程和解答为一个完整的文本
                problem = data.get("problem", "")
                thinking = data.get("thinking", "")
                solution = data.get("solution", "")
                combined_text = f"problem: {problem}, thinking: {thinking}, solution: {solution}"
                original_input_texts.append(combined_text)
        
        print(f"从 {jsonl_filename} 读取了 {len(original_input_texts)} 个文本")
        
        if not resume:
            input_texts = original_input_texts[rank::num_producers]
            sentence_indices = list(range(rank, len(original_input_texts), num_producers))
        else:
            input_texts = []
            sentence_indices = []
            for sentence_idx in range(rank, len(original_input_texts), num_producers):
                if not os.path.exists(f"{p_save_path}/{sentence_idx}"):
                    input_texts.append(original_input_texts[sentence_idx])
                    sentence_indices.append(sentence_idx)
    else:
        # 原始的pickle文件读取逻辑
        with open(dataset_filename, "rb") as f:
            data = pickle.load(f)
            original_input_texts = data["sentences"]
            num_sentences = len(original_input_texts)
            if not resume:
                input_texts = original_input_texts[rank::num_producers]
                sentence_indices = list(range(rank, num_sentences, num_producers))
            else:
                input_texts = []
                sentence_indices = []
                for sentence_idx in range(rank, num_sentences, num_producers):
                    if not os.path.exists(f"{p_save_path}/{sentence_idx}"):
                        input_texts.append(original_input_texts[sentence_idx])
                        sentence_indices.append(sentence_idx)

    if len(input_texts) > 0:
        tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer(input_texts, padding=True, truncation=False, return_tensors="pt")

        with torch.no_grad():
            for i in tqdm(range(0, len(inputs["input_ids"]), batch_size), position=rank, desc=f"Producer {rank}"):
                model_output = model(
                    input_ids=inputs["input_ids"][i:i+batch_size].to(model.device),
                    attention_mask=inputs["attention_mask"][i:i+batch_size].to(model.device),
                    output_hidden_states=True,
                )
                batch_hidden_states = torch.stack(model_output.hidden_states[1:]).cpu().numpy()  # layer activations (num_layers, B, L, D)
                batch_hidden_states = batch_hidden_states[layer_list]
                batch_attention_mask = inputs["attention_mask"][i:i+batch_size].numpy()  # (B, L)
                actual_batch_size = batch_hidden_states.shape[1]
                batch_sentence_indices = sentence_indices[i:i+actual_batch_size]
                queue.put((batch_hidden_states, batch_attention_mask, batch_sentence_indices))


def run_corr(queue, layer_list, p_save_path, worker_idx, network_density=1.0):
    from torch_geometric.utils import dense_to_sparse
    with torch.no_grad():
        while True:
            batch = queue.get(block=True)
            if batch == "STOP":
                break
            hidden_states, attention_mask, sentence_indices = batch
            for i, sentence_idx in enumerate(sentence_indices):
                for j, layer_idx in enumerate(layer_list):
                    layer_hidden_states = hidden_states[j, i]
                    sentence_attention_mask = attention_mask[i]
                    sentence_hidden_states = layer_hidden_states[sentence_attention_mask == 1].T
                    corr = np.corrcoef(sentence_hidden_states)
                    p_dir_name = f"{p_save_path}/{sentence_idx}"
                    os.makedirs(p_dir_name, exist_ok=True)
                    if network_density < 1.0:
                        percentile_threshold = network_density * 100
                        threshold = np.percentile(np.abs(corr), 100 - percentile_threshold)
                        corr[np.abs(corr) < threshold] = 0
                        np.fill_diagonal(corr, 1.0)
                        corr = torch.from_numpy(corr)
                        edge_index, edge_attr = dense_to_sparse(corr)
                        edge_index = edge_index.numpy()
                        edge_attr = edge_attr.numpy()
                        np.save(f"{p_dir_name}/layer_{layer_idx}_sparse_{network_density}_edge_index.npy", edge_index)
                        np.save(f"{p_dir_name}/layer_{layer_idx}_sparse_{network_density}_edge_attr.npy", edge_attr)
                    else:
                        np.save(f"{p_dir_name}/layer_{layer_idx}_corr.npy", corr)

    print(f"Worker {worker_idx} finished processing.")


def main(_):
    model_name = FLAGS.llm_model_name
    if FLAGS.ckpt_step == -1:
        dir_name = f"data/graph_probing/{model_name}"
    else:
        dir_name = f"data/graph_probing/{model_name}_step{FLAGS.ckpt_step}"
    os.makedirs(dir_name, exist_ok=True)

    layer_list = FLAGS.llm_layer
    queue = Queue()
    producers = []
    hf_model_name = model_name
    for i, gpu_id in enumerate(FLAGS.gpu_id):
        p = Process(
            target=run_llm,
            args=(
                i,
                len(FLAGS.gpu_id),
                queue,
                FLAGS.dataset_filename,
                hf_model_name,
                FLAGS.ckpt_step,
                gpu_id,
                FLAGS.batch_size,
                layer_list,
                FLAGS.resume,
                dir_name,
                FLAGS.use_jsonl,
                FLAGS.jsonl_filename,
            )
        )
        p.start()
        producers.append(p)

    num_workers = FLAGS.num_workers
    consumers = []
    for worker_idx in range(num_workers):
        p = Process(
            target=run_corr,
            args=(queue, layer_list, dir_name, worker_idx, FLAGS.network_density))
        p.start()
        consumers.append(p)

    for producer in producers:
        producer.join()
    for _ in range(num_workers):
        queue.put("STOP")
    for consumer in consumers:
        consumer.join()


if __name__ == "__main__":
    setproctitle("debug@zhengyu")
    app.run(main)
