import argparse

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


def calculate_metrics(query_vectors, code_vectors, recall_k=5, mrr_k=10):
    search_k = max(recall_k, mrr_k)
    index = faiss.IndexFlatL2(code_vectors.shape[1])
    index.add(code_vectors)

    hits = 0
    reciprocal_ranks = []

    for i in tqdm(range(len(query_vectors)), desc="Evaluating"):
        query = query_vectors[i].reshape(1, -1)
        _, indices = index.search(query, search_k)
        retrieved = indices[0]

        if i in retrieved[:recall_k]:
            hits += 1

        rank_positions = np.where(retrieved[:mrr_k] == i)[0]
        if len(rank_positions) > 0:
            reciprocal_ranks.append(1.0 / (rank_positions[0] + 1))
        else:
            reciprocal_ranks.append(0.0)

    recall = hits / len(query_vectors)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return recall, mrr


def main():
    parser = argparse.ArgumentParser(description="Evaluate GraphCodeBERT code search on CodeSearchNet.")
    parser.add_argument("--language", default="python", help="CodeSearchNet language subset.")
    parser.add_argument("--split", default="train[:10000]", help="Dataset slice for evaluation.")
    args = parser.parse_args()

    print(f"Loading evaluation set ({args.language}, {args.split})...")
    dataset = load_dataset("code_search_net", args.language, split=args.split)

    queries = [item["func_documentation_string"] or "" for item in dataset]
    codes = [item["func_code_string"] or "" for item in dataset]

    tokenizer, model, device = load_model()

    print("Encoding docstrings...")
    query_vectors = encode_texts(queries, tokenizer, model, device)

    print("Encoding code...")
    code_vectors = encode_texts(codes, tokenizer, model, device)

    recall_at_5, mrr_at_10 = calculate_metrics(query_vectors, code_vectors)

    print("\nClosed-set retrieval results")
    print(f"Recall@5: {recall_at_5:.4f}")
    print(f"MRR@10:   {mrr_at_10:.4f}")


if __name__ == "__main__":
    main()
