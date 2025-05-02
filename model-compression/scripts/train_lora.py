import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import torch


def fine_tune_lora(model_name, data_path, batch_size, iters):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q", "v"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    model = get_peft_model(model, config)

    dataset = load_dataset("json", data_files=data_path, split="train")

    def preprocess(example):
        inputs = tokenizer(example['text'], truncation=True, padding="max_length", max_length=128)
        inputs['labels'] = inputs['input_ids'].copy()
        return inputs

    tokenized = dataset.map(preprocess, batched=True)

    training_args = TrainingArguments(
        output_dir="./lora-output",
        per_device_train_batch_size=batch_size,
        max_steps=iters,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=500,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
    )

    trainer.train()
    model.save_pretrained("./lora-output")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--iters", type=int, default=1000)
    args = parser.parse_args()

    fine_tune_lora(
        model_name=args.model,
        data_path=args.data,
        batch_size=args.batch_size,
        iters=args.iters
    )
#     print("📦 Loading model...")