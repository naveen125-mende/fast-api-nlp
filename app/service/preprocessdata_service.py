import json
from transformers import PegasusTokenizer

class PreProcessData:
    @staticmethod
    def preprocess_data(req):
        # Load the dataset from JSON file
        with open("app/data/job_summarize.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        # Separate inputs and summaries from the list of dicts
        inputs_texts = [item["input"] for item in data]
        summaries_texts = [item["summary"] for item in data]

        # Load tokenizer
        tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")

        # Tokenize inputs and summaries
        inputs = tokenizer(inputs_texts, truncation=True, padding="longest", max_length=req.max_length)
        labels = tokenizer(summaries_texts, truncation=True, padding="longest", max_length=64)

        # Save tokenized data
        with open("app/data/processed.json", "w", encoding="utf-8") as f:
            json.dump({"input_ids": inputs["input_ids"], "labels": labels["input_ids"]}, f)

        return len(inputs["input_ids"])
