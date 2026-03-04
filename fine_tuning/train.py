import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from fine_tuning.dataset import load_dataset

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def main():
    # Load dataset
    dataset = load_dataset("data/finetune_data.json")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # IMPORTANT: set padding token
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(example):
        tokens = tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=256
        )

        # Add labels (CRITICAL)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized_dataset = dataset.map(tokenize_function)

    # Remove unused columns
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])

    # LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir="./fine_tuned_model",
        per_device_train_batch_size=2,
        num_train_epochs=2,
        logging_steps=1,
        save_strategy="epoch",
        learning_rate=2e-4,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset
    )

    trainer.train()

    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")


if __name__ == "__main__":
    main()