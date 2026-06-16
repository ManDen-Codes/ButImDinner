import os

import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)


def load_config():
    config_path = os.path.join(ROOT_DIR, "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    tc = config.get("training", {})
    base_model = tc.get("base_model", "unsloth/Qwen3-4B")

    adapter_path = os.path.join(ROOT_DIR, "data", "model", "lora_adapter")
    merged_path = os.path.join(ROOT_DIR, "data", "model", "merged")
    os.makedirs(merged_path, exist_ok=True)

    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    print(f"Loading LoRA adapter from {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("Merging adapter into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {merged_path}")
    model.save_pretrained(merged_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_path)

    print("\nMerge complete. Now convert to GGUF:")
    print("  git clone https://github.com/ggerganov/llama.cpp")
    print("  pip install -r llama.cpp/requirements.txt")
    print(f"  python llama.cpp/convert_hf_to_gguf.py {merged_path} --outfile data/model/dinner_f16.gguf --outtype f16")
    print("  llama.cpp/build/bin/llama-quantize data/model/dinner_f16.gguf data/model/dinner.gguf Q4_K_M")


if __name__ == "__main__":
    main()
