# 🔧 Model Compression Techniques for Large Language Models (LLMs)

Welcome to the **Model Compression for LLMs** repository! This repo implements and compares key compression techniques that enable large-scale language models to run efficiently on consumer hardware (e.g., laptops, edge devices) and with minimal computational cost.

---

## 📌 Why Model Compression?
Modern LLMs have billions of parameters, making them expensive to store, fine-tune, and deploy. Compression helps by:
- Reducing memory usage (RAM/VRAM)
- Lowering inference latency
- Enabling training/fine-tuning on limited hardware

---

## 🧰 Techniques Implemented

| Technique         | Description                                             | Directory       |
|------------------|---------------------------------------------------------|-----------------|
| **LoRA**         | Low-Rank Adaptation: fine-tunes only adapter layers     | `scripts/train_lora.py`, `notebooks/1_LoRA_FineTuning.ipynb` |
| **Quantization** | Reduces weight precision (e.g., 8-bit, 4-bit)           | `scripts/quantize_post_train.py`, `notebooks/2_Quantization.ipynb` |
| **Distillation** | Trains a small "student" model to mimic a larger one    | `scripts/train_distill.py`, `notebooks/3_Distillation.ipynb` |
| **Pruning**      | Removes unimportant weights or neurons from the model   | `scripts/prune_model.py`, `notebooks/4_Pruning.ipynb` |

---

## 🚀 Quick Start

### 1. Clone the repo & install dependencies
```bash
pip install -r requirements.txt
```

### 2. Fine-tune with LoRA (example)
```bash
python scripts/train_lora.py \
    --model google/flan-t5-small \
    --data ./data/sample_train.jsonl \
    --batch-size 4 --iters 500
```

### 3. Quantize a model (post-training)
```bash
python scripts/quantize_post_train.py \
    --model ./models/flan-t5-small \
    --output ./models/flan-quantized
```

### 4. Evaluate a base model:
```bash
python scripts/evaluate_compression.py \
    --model google/flan-t5-small \
    --prompt "What is the capital of France?"
```
### 5. Evaluate a LoRA-adapted model:
```bash
python scripts/evaluate_compression.py \
    --model google/flan-t5-small \
    --prompt "Explain quantum computing" \
    --lora lora-output/
```


---

## 📊 Evaluation
All techniques are evaluated using:
- Accuracy / BLEU / F1 (task-specific)
- Model size
- Inference speed (tokens/sec)
- Memory usage

See `notebooks/5_Evaluation.ipynb` for benchmarking.

---

## 🤝 Contributions
Pull requests are welcome! To add a new method or model:
- Place reusable code in `compression/`
- Add script in `scripts/`
- Optionally add demo notebook

---

## 📄 License
MIT License

