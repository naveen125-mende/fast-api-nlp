import os
import torch
from datasets import load_dataset
from transformers import PegasusTokenizer, PegasusForConditionalGeneration, Trainer, TrainingArguments
from fastapi import HTTPException


class TrainerService:
    @staticmethod
    async def train_model(request):
        try:
            abs_path = os.path.abspath(request.train_file)

            if not os.path.exists(abs_path):
                raise HTTPException(status_code=400, detail=f"❌ Dataset file not found at: {abs_path}")

            # Load dataset
            dataset = load_dataset('json', data_files={'train': abs_path})
            tokenizer = PegasusTokenizer.from_pretrained(request.model_name)
            model = PegasusForConditionalGeneration.from_pretrained(request.model_name)

            # Preprocessing function
            def preprocess(examples):
                model_inputs = tokenizer(
                    examples["input"], max_length=512, truncation=True, padding='max_length'
                )
                with tokenizer.as_target_tokenizer():
                    labels = tokenizer(
                        examples["summary"], max_length=128, truncation=True, padding='max_length'
                    )
                label_ids = [
                    [(token if token != tokenizer.pad_token_id else -100) for token in label]
                    for label in labels["input_ids"]
                ]
                model_inputs["labels"] = label_ids
                return model_inputs

            # Tokenize dataset
            tokenized_dataset = dataset["train"].map(
                preprocess,
                batched=True,
                remove_columns=dataset["train"].column_names,
            )
            tokenized_dataset.set_format(type="torch")

            # Training arguments
            training_args = TrainingArguments(
                output_dir=request.output_dir,
                per_device_train_batch_size=request.batch_size,
                num_train_epochs=request.num_train_epochs,
                logging_steps=10,
                save_steps=500,
                save_total_limit=2,
                fp16=torch.cuda.is_available(),
                remove_unused_columns=False,
                push_to_hub=False,
            )

            # Trainer setup
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                tokenizer=tokenizer,
            )

            trainer.train()

            model.save_pretrained(request.output_dir)
            tokenizer.save_pretrained(request.output_dir)

            return {
                "success": True,
                "message": "✅ Training complete",
                "output_dir": request.output_dir
            }

        except HTTPException as http_ex:
            raise http_ex
        except Exception as e:
            return {
                "success": False,
                "message": f"❌ Training failed: {str(e)}"
            }
