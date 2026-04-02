import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer


MODEL_NAME = "microsoft/graphcodebert-base"
MAX_LENGTH = 256
BATCH_SIZE = 16


def load_model(model_name: str = MODEL_NAME):
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device


def encode_texts(texts, tokenizer, model, device, batch_size=BATCH_SIZE, max_length=MAX_LENGTH):
    vectors = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs).last_hidden_state
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            masked_outputs = outputs * attention_mask
            pooled = masked_outputs.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)

        vectors.append(pooled.cpu().numpy().astype("float32"))

    return np.vstack(vectors)


def build_indices(vectors: np.ndarray):
    index_l2 = faiss.IndexFlatL2(vectors.shape[1])
    index_l2.add(vectors)

    normalized_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)
    index_ip = faiss.IndexFlatIP(vectors.shape[1])
    index_ip.add(normalized_vectors)

    return index_l2, index_ip


def main():
    parser = argparse.ArgumentParser(description="Build FAISS indices for GraphCodeBERT code search.")
    parser.add_argument("--language", default="python", help="CodeSearchNet language subset.")
    parser.add_argument("--split", default="train", help="Dataset split, e.g. train or train[:10000].")
    parser.add_argument("--output-dir", default="data", help="Directory for vectors, index, and code list.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading CodeSearchNet ({args.language}, {args.split})...")
    dataset = load_dataset("code_search_net", args.language, split=args.split)
    code_list = [item["func_code_string"] for item in dataset if item.get("func_code_string")]
    print(f"Loaded {len(code_list)} code snippets.")

    tokenizer, model, device = load_model()
    vectors = encode_texts(code_list, tokenizer, model, device)

    np.save(output_dir / "code_vectors.npy", vectors)
    with open(output_dir / "code_list.json", "w", encoding="utf-8") as f:
        json.dump(code_list, f, ensure_ascii=False)

    index_l2, index_ip = build_indices(vectors)
    faiss.write_index(index_l2, str(output_dir / "codesearch_l2.faiss"))
    faiss.write_index(index_ip, str(output_dir / "codesearch_ip.faiss"))

    print("Saved:")
    print(f"- {output_dir / 'code_vectors.npy'}")
    print(f"- {output_dir / 'code_list.json'}")
    print(f"- {output_dir / 'codesearch_l2.faiss'}")
    print(f"- {output_dir / 'codesearch_ip.faiss'}")


if __name__ == "__main__":
    main()
