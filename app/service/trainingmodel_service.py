import json
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, Trainer, TrainingArguments

class TrainingModel:
    def train_model():
        with open("app/data/processed.json", "r", encoding="utf-8") as f:
            processed_data = json.load(f)

        inputs = processed_data["input_ids"]
        labels = processed_data["labels"]

        train_data = [
            {"input_ids": torch.tensor(inp, dtype=torch.long),
            "labels": torch.tensor(lbl, dtype=torch.long)}
            for inp, lbl in zip(inputs, labels)
        ]

        model_name = "google/pegasus-xsum"
        tokenizer = PegasusTokenizer.from_pretrained(model_name)
        model = PegasusForConditionalGeneration.from_pretrained(model_name)

        training_args = TrainingArguments(
            output_dir="./pegasus_finetuned",
            per_device_train_batch_size=2,
            num_train_epochs=3,
            logging_dir="./logs",
            logging_steps=5,
            no_cuda=False  # Set True if you don't have GPU
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            tokenizer=tokenizer,
        )

        # Start training (blocking)
        trainer.train()

        return {"epochs": 3, "batch_size": 2, "trained_samples": len(train_data)}
