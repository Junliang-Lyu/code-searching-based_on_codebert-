# %%
from transformers import RobertaTokenizer, RobertaModel
import torch

# 加载 GraphCodeBERT 模型
tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
model = RobertaModel.from_pretrained("microsoft/graphcodebert-base")
model.eval()  # 关闭dropout


# %%
import numpy
import torch
import transformers
import datasets

print(datasets.__version__)
print("NumPy:", numpy.__version__)
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)


# %%
from datasets import load_dataset

# 加载原始 CodeSearchNet Python 子集的训练集
dataset = load_dataset("code_search_net", "python", split="train")


# %%
import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm
from datasets import load_dataset

# 保存路径
output_vectors_path = r"C:\Users\ASUS\Desktop\summer code\research\data\codesearchnet_vectors.npy"

# 加载 GraphCodeBERT 模型
tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
model = RobertaModel.from_pretrained("microsoft/graphcodebert-base")
model.eval()
device = torch.device('cpu')
model.to(device)

# 定义向量生成函数
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)  
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy() 

# 加载 CodeSearchNet 原始数据集（Python子集）
print("正在加载 CodeSearchNet 数据集（Python子集）...")
dataset = load_dataset("code_search_net", "python", split="train")

print("正在提取代码片段...")
code_list = [item['func_code_string'] for item in dataset]
print(f"共加载 {len(code_list)} 个代码片段")

# 批量生成向量
vectors = []

print("正在生成向量...（仅CPU）")
for code in tqdm(code_list):
    vector = get_embedding(code)
    vectors.append(vector)

vectors = np.array(vectors)

# 保存向量
np.save(output_vectors_path, vectors)
print(f"向量生成完成！已保存为 {output_vectors_path}")


# %%
import faiss
import numpy as np
 
# ====== 配置路径 ======
vector_path = r"C:\Users\ASUS\Desktop\summer code\research\data\codesearchnet_vectors.npy"
output_index_L2_path = r"C:\Users\ASUS\Desktop\summer code\research\data\codesearchnet_index_L2.faiss"
output_index_IP_path = r"C:\Users\ASUS\Desktop\summer code\research\data\codesearchnet_index_IP.faiss"

# ====== 加载向量文件 ======
vectors = np.load(vector_path).astype('float32')
print(f"✅ 已加载向量文件，形状：{vectors.shape}")

# ====== 建立 L2 距离索引 ======
index_L2 = faiss.IndexFlatL2(vectors.shape[1])
index_L2.add(vectors)
faiss.write_index(index_L2, output_index_L2_path)
print(f"✅ L2 索引已保存至：{output_index_L2_path}")

# ====== 建立 余弦相似度（内积）索引 ======
vectors_IP = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
index_IP = faiss.IndexFlatIP(vectors.shape[1])
index_IP.add(vectors_IP)
faiss.write_index(index_IP, output_index_IP_path)
print(f"✅ 余弦相似度索引已保存至：{output_index_IP_path}")


# %%
from datasets import load_dataset
import json

# 加载完整训练集
dataset = load_dataset("code_search_net", "python", split="train")

# 提取代码段
code_list = [item['func_code_string'] for item in dataset]

# 保存为 JSON
with open(r"C:\Users\ASUS\Desktop\summer code\research\data\codesearchnet_code_list.json", "w", encoding="utf-8") as f:
    json.dump(code_list, f)

print("✅ 已保存完整 code_list，共", len(code_list), "条")


# %%
import numpy as np
import json

vecs = np.load(r"C:\Users\ASUS\Desktop\summer code\research\data\codesearchnet_vectors.npy")
print("向量数量：", vecs.shape[0])

with open(r"C:\Users\ASUS\Desktop\summer code\research\data\codesearchnet_code_list.json", "r", encoding="utf-8") as f:
    codes = json.load(f)
print("代码段数量：", len(codes))


# %%
import tkinter as tk
from tkinter import scrolledtext
import numpy as np
import faiss
from transformers import RobertaTokenizer, RobertaModel
import torch
import json

# ===== Load FAISS Indices =====
index_L2 = faiss.read_index(r"C:\Users\ASUS\Desktop\summer code\research\data\codesearchnet_index_L2.faiss")
index_IP = faiss.read_index(r"C:\Users\ASUS\Desktop\summer code\research\data\codesearchnet_index_IP.faiss")

# ===== Load code vectors =====
code_vectors = np.load(r"C:\Users\ASUS\Desktop\summer code\research\data\codesearchnet_vectors.npy")

# ===== Load code list (aligned with vectors) =====
with open(r"C:\Users\ASUS\Desktop\summer code\research\data\codesearchnet_code_list.json", "r", encoding="utf-8") as f:
    code_list = json.load(f)

assert len(code_vectors) == len(code_list), "❌ Mismatch between code list and vector count!"

# ===== Load pretrained model =====
tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
model = RobertaModel.from_pretrained("microsoft/graphcodebert-base")
model.eval()

# ===== Vector encoding function =====
def get_query_vector(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy().astype('float32')

# ===== Search and display results =====
def search_and_display():
    query = query_entry.get()
    if not query.strip():
        result_box.insert(tk.END, "Please enter a query.\n")
        return

    query_vector = get_query_vector(query).reshape(1, -1)

    if search_mode.get() == "L2":
        D, I = index_L2.search(query_vector, 5)
    else:
        normalized_query = query_vector / np.linalg.norm(query_vector)
        D, I = index_IP.search(normalized_query, 5)

    result_box.delete(1.0, tk.END)

    for rank, idx in enumerate(I[0]):
        if idx >= len(code_list):
            result_box.insert(tk.END, f"Index {idx} is out of range. Skipped.\n")
            continue

        code_snippet = code_list[idx]
        vector_candidate = code_vectors[idx]

        # Local scoring
        l2_distance = np.linalg.norm(query_vector.flatten() - vector_candidate.flatten())
        norm_query = query_vector / np.linalg.norm(query_vector)
        norm_candidate = vector_candidate / np.linalg.norm(vector_candidate)
        cosine_sim = float(np.dot(norm_query.flatten(), norm_candidate.flatten()))

        result_box.insert(tk.END, f"[Top {rank+1}]  L2: {l2_distance:.4f}    Cosine: {cosine_sim:.4f}\n")
        result_box.insert(tk.END, code_snippet + "\n\n")

# ===== Tkinter GUI =====
window = tk.Tk()
window.title("Code Search Engine")
window.geometry("1200x800")

tk.Label(window, text="Enter your query (function description / keywords):").pack(pady=10)
query_entry = tk.Entry(window, width=120)
query_entry.pack(pady=5)

search_mode = tk.StringVar()
search_mode.set("L2")
mode_frame = tk.Frame(window)
tk.Label(mode_frame, text="Search metric:").pack(side=tk.LEFT)
tk.Radiobutton(mode_frame, text="L2 Distance", variable=search_mode, value="L2").pack(side=tk.LEFT)
tk.Radiobutton(mode_frame, text="Cosine Similarity", variable=search_mode, value="Cosine").pack(side=tk.LEFT)
mode_frame.pack(pady=10)

tk.Button(window, text="Search Code", command=search_and_display).pack(pady=10)
result_box = scrolledtext.ScrolledText(window, width=140, height=35)
result_box.pack(pady=10)

window.mainloop()


# %%
import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
from datasets import load_dataset
from tqdm import tqdm
import faiss

# ✅ 加载模型
tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
model = RobertaModel.from_pretrained("microsoft/graphcodebert-base")
model.eval()
device = torch.device("cpu")
model.to(device)

# ✅ 定义编码函数（mean pooling）
def encode(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().astype('float32')

# ✅ 加载训练集前10000条
dataset = load_dataset("code_search_net", "python", split="train[:10000]")
queries = [item['func_documentation_string'] for item in dataset]
codes   = [item['func_code_string'] for item in dataset]


# ✅ 生成向量（query: docstring, db: code）
print("✅ 正在编码 docstring（query）向量...")
query_vecs = np.array([encode(q) for q in tqdm(queries)])

print("✅ 正在编码 code（候选）向量...")
code_vecs = np.array([encode(c) for c in tqdm(codes)])

# ✅ 构建索引（L2）
index = faiss.IndexFlatL2(code_vecs.shape[1])
index.add(code_vecs)

# ✅ 评估 Recall@5 和 MRR
recall_at_5 = 0
reciprocal_ranks = []

print("\n✅ 正在执行闭集检索与评估...")
for i in range(len(query_vecs)):
    q = query_vecs[i].reshape(1, -1)
    _, I = index.search(q, 5)
    retrieved_indices = I[0]

    if i in retrieved_indices:
        recall_at_5 += 1
        rank = np.where(retrieved_indices == i)[0][0] + 1
        reciprocal_ranks.append(1.0 / rank)
    else:
        reciprocal_ranks.append(0.0)

# ✅ 输出结果
mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
recall = recall_at_5 / len(query_vecs)

print(f"\n🎯 闭集评估结果（前10000样本）")
print(f"MRR: {mrr:.4f}")
print(f"Recall@5: {recall:.4f}")



