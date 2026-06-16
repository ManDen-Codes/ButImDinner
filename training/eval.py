import os
import sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import yaml
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)


def load_config():
    config_path = os.path.join(ROOT_DIR, "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def format_chat(example, tokenizer, max_seq_length):
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False,
        enable_thinking=False,
    )
    tokens = tokenizer(text, truncation=True, max_length=max_seq_length, padding=False)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def main():
    config = load_config()
    tc = config.get("training", {})

    base_model = tc.get("base_model", "Qwen/Qwen3.5-9B")
    max_seq_length = tc.get("max_seq_length", 768)

    adapter_dir = os.path.join(ROOT_DIR, "data", "model", "lora_adapter")

    # Allow passing a specific checkpoint as argument
    if len(sys.argv) > 1:
        checkpoint = sys.argv[1]
    else:
        checkpoints = sorted(
            [d for d in os.listdir(adapter_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1]),
        )
        if not checkpoints:
            print("No checkpoints found.")
            return
        checkpoint = os.path.join(adapter_dir, checkpoints[-1])

    print(f"Evaluating checkpoint: {checkpoint}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, checkpoint)
    model.eval()

    val_path = os.path.join(ROOT_DIR, "data", "processed", "val.jsonl")
    print("Loading val dataset...")
    dataset = load_dataset("json", data_files={"validation": val_path})
    dataset = dataset.map(
        lambda ex: format_chat(ex, tokenizer, max_seq_length),
        remove_columns=dataset["validation"].column_names,
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="/tmp/eval_tmp",
            per_device_eval_batch_size=1,
            eval_accumulation_steps=16,
            bf16=True,
            report_to="none",
        ),
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    print("Running eval...")
    metrics = trainer.evaluate()
    print(f"\nEval loss: {metrics['eval_loss']:.4f}")
    print(f"Perplexity: {torch.exp(torch.tensor(metrics['eval_loss'])):.2f}")


if __name__ == "__main__":
    main()
