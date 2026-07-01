import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
import torch  # MUST be the last import (datasets crashes silently on Windows otherwise)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)


def load_config():
    with open(os.path.join(ROOT_DIR, "config.yaml")) as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    rc = config.get("reaction", {})

    # Tiny base model that decides Dinner's reaction: context -> ":emoji:" or "none".
    # Runs as a SECOND model in the bot ("cluster"), output constrained by a GBNF
    # grammar to Dinner's emoji set + "none". Small enough to skip 4-bit quant.
    base_model = rc.get("base_model", "Qwen/Qwen3-0.6B-Base")
    max_seq_length = rc.get("max_seq_length", 768)
    lora_rank = rc.get("lora_rank", 32)
    lora_alpha = rc.get("lora_alpha", 64)
    epochs = rc.get("epochs", 4)
    batch_size = rc.get("batch_size", 8)
    grad_accum = rc.get("gradient_accumulation_steps", 2)
    lr = rc.get("learning_rate", 2e-4)

    print(f"Loading reaction base model: {base_model}")

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = max_seq_length

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

    train_path = os.path.join(ROOT_DIR, "data", "processed", "reactions_train.jsonl")
    val_path = os.path.join(ROOT_DIR, "data", "processed", "reactions_val.jsonl")

    print("Loading reaction dataset...")
    # Rows: {"prompt": "<transcript>\n[reaction]:", "completion": " :emoji:" | " none"}.
    dataset = load_dataset("json", data_files={"train": train_path, "validation": val_path})

    output_dir = os.path.join(ROOT_DIR, "data", "model", "reaction_adapter")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=SFTConfig(
            output_dir=output_dir,
            max_length=max_seq_length,
            completion_only_loss=True,   # loss only on the emoji/none decision
            packing=False,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            num_train_epochs=epochs,
            learning_rate=lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            bf16=True,
            logging_steps=10,
            eval_strategy="epoch",       # tiny eval set; safe to run inline here
            save_strategy="epoch",
            save_total_limit=3,
            # Sparse/imbalanced reaction data overfits fast (eval_loss bottoms ~epoch 2
            # then climbs). Keep the best-generalizing checkpoint, not the last one.
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            seed=42,
        ),
    )

    print("Starting reaction training...")
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

    print(f"Saving reaction adapter to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Reaction training complete!")


if __name__ == "__main__":
    main()
