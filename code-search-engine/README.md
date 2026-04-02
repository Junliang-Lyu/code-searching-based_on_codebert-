# GraphCodeBERT Code Search Engine

A lightweight semantic code search demo built with GraphCodeBERT and FAISS for natural-language-to-code retrieval.

## Overview

This project implements a local code search pipeline for Python code. It encodes code snippets with GraphCodeBERT, builds FAISS indices for retrieval, and provides both evaluation scripts and a simple local demo interface.

The project was built as a portfolio / research-style implementation focused on code retrieval, indexing, and ranking.

## Features

- GraphCodeBERT-based code embeddings
- FAISS indexing for fast similarity search
- Local search demo via `app.py`
- Evaluation pipeline for retrieval quality
- Supports large-scale retrieval over approximately **300,000 code snippets**

## Project Structure

```text
.
├── app.py
├── build_index.py
├── evaluate.py
├── requirements.txt
├── sample_queries.txt
├── assets/
│   └── put_demo_screenshot_here.txt
└── README.md
```

## Dataset and Model Source

This project uses the **Python subset of CodeSearchNet** for retrieval experiments.

The model implementation is based on **GraphCodeBERT** from the Microsoft **CodeBERT** repository. The full dataset and generated vector/index artifacts are intentionally excluded from this repository because they are too large for a clean GitHub demo.

Example dataset loading:

```python
from datasets import load_dataset

dataset = load_dataset("code_search_net", "python", split="train")
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Build embeddings and FAISS indices:

```bash
python build_index.py --language python --split train --output-dir data
```

Run evaluation on a smaller slice:

```bash
python evaluate.py --language python --split "train[:10000]"
```

Launch the local demo:

```bash
python app.py
```

## Results

Current project results:

- **MRR@10 = 0.40**
- **Recall@5 = 0.45**
- Retrieval corpus size: **300,000 code snippets**

## Notes

- This version uses mean pooling over the final hidden states.
- Building the full index on CPU can take a long time.
- The GUI is intentionally simple and meant as a local demo.
- This repository is intended as a clean GitHub / portfolio version rather than a full research reproduction package.

## Acknowledgements

- Microsoft **CodeBERT / GraphCodeBERT** repository
- **CodeSearchNet** benchmark dataset
