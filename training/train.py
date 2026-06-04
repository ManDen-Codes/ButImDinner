import os

import yaml
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)


def load_config():
    config_path = os.path.join(ROOT_DIR, "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def format_chat(example, tokenizer):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
    }


def main():
    config = load_config()
    tc = config.get("training", {})

    base_model = tc.get("base_model", "unsloth/Qwen3.5-4B")
    max_seq_length = tc.get("max_seq_length", 2048)
    lora_rank = tc.get("lora_rank", 64)
    lora_alpha = tc.get("lora_alpha", 128)
    epochs = tc.get("epochs", 3)
    batch_size = tc.get("batch_size", 2)
    grad_accum = tc.get("gradient_accumulation_steps", 4)
    lr = tc.get("learning_rate", 2e-4)

    print(f"Loading base model: {base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    train_path = os.path.join(ROOT_DIR, "data", "processed", "train.jsonl")
    val_path = os.path.join(ROOT_DIR, "data", "processed", "val.jsonl")

    print("Loading dataset...")
    dataset = load_dataset(
        "json", data_files={"train": train_path, "validation": val_path}
    )

    dataset = dataset.map(
        lambda ex: format_chat(ex, tokenizer),
        remove_columns=dataset["train"].column_names,
    )

    output_dir = os.path.join(ROOT_DIR, "data", "model", "lora_adapter")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            num_train_epochs=epochs,
            learning_rate=lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            fp16=True,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            seed=42,
            max_seq_length=max_seq_length,
        ),
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving LoRA adapter to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training complete!")


if __name__ == "__main__":
    main()
