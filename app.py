import json
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, scrolledtext

import faiss
import numpy as np
import torch
from transformers import RobertaModel, RobertaTokenizer


MODEL_NAME = "microsoft/graphcodebert-base"
MAX_LENGTH = 256
DATA_DIR = Path("data")
TOP_K = 5


class CodeSearchApp:
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
        self.model = RobertaModel.from_pretrained(MODEL_NAME)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.code_vectors = np.load(DATA_DIR / "code_vectors.npy").astype("float32")
        with open(DATA_DIR / "code_list.json", "r", encoding="utf-8") as f:
            self.code_list = json.load(f)

        self.index_l2 = faiss.read_index(str(DATA_DIR / "codesearch_l2.faiss"))
        self.index_ip = faiss.read_index(str(DATA_DIR / "codesearch_ip.faiss"))

        if len(self.code_vectors) != len(self.code_list):
            raise ValueError("Vector count does not match code list length.")

        self.window = tk.Tk()
        self.window.title("Code Search Engine")
        self.window.geometry("1200x800")
        self.mode_var = tk.StringVar(value="Cosine")
        self._build_ui()

    def encode_query(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs).last_hidden_state
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            masked_outputs = outputs * attention_mask
            pooled = masked_outputs.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)

        return pooled.squeeze(0).cpu().numpy().astype("float32")

    def search(self):
        query = self.query_entry.get().strip()
        if not query:
            messagebox.showinfo("Missing query", "Please enter a query first.")
            return

        query_vector = self.encode_query(query).reshape(1, -1)
        mode = self.mode_var.get()

        if mode == "L2":
            scores, indices = self.index_l2.search(query_vector, TOP_K)
        else:
            normalized_query = query_vector / np.linalg.norm(query_vector, axis=1, keepdims=True).clip(min=1e-12)
            scores, indices = self.index_ip.search(normalized_query, TOP_K)

        self.result_box.delete("1.0", tk.END)

        for rank, idx in enumerate(indices[0], start=1):
            snippet = self.code_list[idx]
            if mode == "L2":
                header = f"[Top {rank}] L2 distance: {scores[0][rank - 1]:.4f}"
            else:
                header = f"[Top {rank}] Cosine similarity: {scores[0][rank - 1]:.4f}"

            self.result_box.insert(tk.END, header + "\n")
            self.result_box.insert(tk.END, snippet + "\n\n")

    def _build_ui(self):
        tk.Label(
            self.window,
            text="Enter a natural language description or keyword query:",
        ).pack(pady=10)

        self.query_entry = tk.Entry(self.window, width=120)
        self.query_entry.pack(pady=5)

        mode_frame = tk.Frame(self.window)
        tk.Label(mode_frame, text="Search metric:").pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(mode_frame, text="Cosine Similarity", variable=self.mode_var, value="Cosine").pack(side=tk.LEFT)
        tk.Radiobutton(mode_frame, text="L2 Distance", variable=self.mode_var, value="L2").pack(side=tk.LEFT)
        mode_frame.pack(pady=10)

        tk.Button(self.window, text="Search Code", command=self.search).pack(pady=10)

        self.result_box = scrolledtext.ScrolledText(self.window, width=140, height=35)
        self.result_box.pack(pady=10)

    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    app = CodeSearchApp()
    app.run()
