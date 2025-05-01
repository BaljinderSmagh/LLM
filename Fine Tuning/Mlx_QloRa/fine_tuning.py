import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import AdamW
from transformers import AutoTokenizer
from pathlib import Path
import json
import argparse
from tqdm import trange

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=16):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.A = nn.Linear(in_features, r, bias=False)
        self.B = nn.Linear(r, out_features, bias=False)
        self.scaling = alpha / r
        self.linear.trainable = False

    def __call__(self, x):
        return self.linear(x) + self.B(self.A(x)) * self.scaling

class MiniLLM(nn.Module):
    def __init__(self, dim, lora_layers=4):
        super().__init__()
        self.layers = [LoRALinear(dim, dim) for _ in range(lora_layers)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def load_jsonl_dataset(path, tokenizer, max_length=128):
    with open(path) as f:
        texts = [json.loads(line)["text"] for line in f]
    tokens = tokenizer(texts, return_tensors="np", padding=True, truncation=True, max_length=max_length)
    return mx.array(tokens["input_ids"])

def train(model, data, iters=100, batch_size=2, lr=1e-4):
    optimizer = AdamW(model.trainable_parameters(), lr=lr)
    for step in trange(iters):
        idx = mx.random.randint(0, data.shape[0] - batch_size)
        x = data[idx:idx + batch_size]
        y = x

        def loss_fn():
            pred = model(x.astype(mx.float32))
            return ((pred - y.astype(mx.float32)) ** 2).mean()

        loss, grads = mx.value_and_grad(loss_fn)()
        optimizer.update(grads)

        if step % 10 == 0:
            print(f"[Step {step}] Loss: {loss.item():.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--lora-layers", type=int, default=4)
    args = parser.parse_args()

    print("📦 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"📂 Loading dataset from {args.data}/train.jsonl...")
    data = load_jsonl_dataset(f"{args.data}/train.jsonl", tokenizer)

    print("🧠 Building model...")
    model = MiniLLM(dim=data.shape[-1], lora_layers=args.lora_layers)

    print(f"🚀 Starting training for {args.iters} iterations...")
    train(model, data, iters=args.iters, batch_size=args.batch_size)

    print("💾 Saving adapter weights...")
    Path("adapters").mkdir(exist_ok=True)
    model.save_weights("adapters/lora_adapters.npz")

if __name__ == "__main__":
    main()
