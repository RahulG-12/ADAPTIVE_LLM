from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_PATH = "./fine_tuned_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=150
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    while True:
        query = input("Enter prompt: ")
        print(generate(query))