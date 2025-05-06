import time
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel,PeftModelForSeq2SeqLM


def evaluate_model(model_path, prompt, use_lora=False, adapter_path=None):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    if use_lora and adapter_path:
        model = PeftModelForSeq2SeqLM.from_pretrained(model, adapter_path)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    start = time.time()
    outputs = model.generate(input_ids=input_ids, max_new_tokens=50)
    latency = time.time() - start

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Response:", response)
    print(f"Latency: {latency:.3f} seconds")
    print(f"Model size: ~{round(sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6, 2)} MB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to model or model name")
    parser.add_argument("--prompt", type=str, help="Prompt to evaluate")
    parser.add_argument("--lora", type=str, default=None, help="Optional: LoRA adapter path")
    args = parser.parse_args()

    evaluate_model(args.model, args.prompt, use_lora=bool(args.lora), adapter_path=args.lora)
