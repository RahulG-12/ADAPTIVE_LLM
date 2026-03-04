import json
from datasets import Dataset

def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    for item in data:
        prompt = f"Instruction: {item['instruction']}\nResponse: {item['response']}"
        texts.append({"text": prompt})

    return Dataset.from_list(texts)