# 📓 Distillation Notebook: Train a Student LLM from a Teacher

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, DataCollatorForSeq2Seq, Trainer
from datasets import load_dataset
import torch
import torch.nn.functional as F
from peft import PeftModel


#  Force CPU for all operations (avoid MPS issues on Mac)
device = torch.device("cpu")

class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model.to(device).eval()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)
        student_outputs = model(**inputs)

        loss = F.kl_div(
            input=F.log_softmax(student_outputs.logits / 1.0, dim=-1),
            target=F.softmax(teacher_outputs.logits / 1.0, dim=-1),
            reduction="batchmean"
        )
        return (loss, student_outputs) if return_outputs else loss


# Define models
teacher_model = "google/flan-t5-base"
student_model = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(student_model)
teacher = AutoModelForSeq2SeqLM.from_pretrained(teacher_model).to(device)
student = AutoModelForSeq2SeqLM.from_pretrained(student_model).to(device)

# 📁 Load and preprocess data
file_path = "./data/sample_train.jsonl"
dataset = load_dataset("json", data_files=file_path, split="train")

def preprocess(example):
    x = tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)
    x["labels"] = x["input_ids"].copy()
    return x

dataset = dataset.map(preprocess, batched=True)

# 🛠️ Training configuration
args = TrainingArguments(
    output_dir="../distilled-student",
    per_device_train_batch_size=2,
    max_steps=100,
    logging_steps=10,
    save_steps=100,
    report_to="none",
    fp16=False,
)

trainer = DistillationTrainer(
    model=student,
    teacher_model=teacher,
    args=args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=student)
)

#  Train the student model
trainer.train()

# Save outputs
trainer.model.save_pretrained("../distilled-student")
tokenizer.save_pretrained("../distilled-student")
print("Training complete! Distilled model saved to ../distilled-student")