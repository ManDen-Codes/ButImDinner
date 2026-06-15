import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)


def load_config():
    config_path = os.path.join(ROOT_DIR, "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def format_chat(example, tokenizer):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )
    }


def main():
    config = load_config()
    tc = config.get("training", {})

    base_model = tc.get("base_model", "unsloth/Qwen3-4B")
    max_seq_length = tc.get("max_seq_length", 512)
    lora_rank = tc.get("lora_rank", 64)
    lora_alpha = tc.get("lora_alpha", 128)
    epochs = tc.get("epochs", 5)
    batch_size = tc.get("batch_size", 4)
    grad_accum = tc.get("gradient_accumulation_steps", 4)
    lr = tc.get("learning_rate", 2e-4)

    print(f"Loading base model: {base_model}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = max_seq_length

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

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
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=SFTConfig(
            output_dir=output_dir,
            dataset_text_field="text",
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            num_train_epochs=epochs,
            learning_rate=lr,
            lr_scheduler_type="cosine",
            warmup_steps=100,
            bf16=True,
            logging_steps=10,
            eval_strategy="no",
            save_strategy="steps",
            save_steps=150,
            save_total_limit=3,
            seed=42,
        ),
    )

    print("Starting training...")
    latest_checkpoint = None
    if os.path.isdir(output_dir):
        checkpoints = sorted(
            [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1]),
        )
        if checkpoints:
            latest_checkpoint = os.path.join(output_dir, checkpoints[-1])
            print(f"Resuming from checkpoint: {latest_checkpoint}")
    trainer.train(resume_from_checkpoint=latest_checkpoint)

    print(f"Saving LoRA adapter to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training complete!")


if __name__ == "__main__":
    main()
