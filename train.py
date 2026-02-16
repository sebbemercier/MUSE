# Copyright 2026 The OpenSLM Project
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

def train_muse(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    print(f"Fine-tuning MUSE pour la rédaction/résumé...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_dataset("json", data_files="data/train_summarization.jsonl", split="train")

    def tokenize_function(examples):
        # Formatage : Document technique -> Résumé SEO
        prompts = [f"Technical Info: {d}
SEO Summary: {s}" for d, s in zip(examples['document'], examples['summary'])]
        return tokenizer(prompts, padding="max_length", truncation=True, max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

    args = TrainingArguments(output_dir="./muse-output", per_device_train_batch_size=4, num_train_epochs=1)
    trainer = Trainer(model=model, args=args, train_dataset=tokenized_datasets)
    trainer.train()
    model.save_pretrained("./fine_tuned_muse")

if __name__ == "__main__":
    print("Script d'entraînement MUSE prêt.")
