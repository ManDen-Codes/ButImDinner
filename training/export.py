"""Merge a LoRA adapter into its base model and print the GGUF conversion command.

  python training/export.py            # persona model  -> data/model/merged
  python training/export.py reaction   # reaction model -> data/model/reaction_merged
"""
import os
import sys

import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)


def load_config():
    with open(os.path.join(ROOT_DIR, "config.yaml")) as f:
        return yaml.safe_load(f)


TARGETS = {
    "persona": {
        "section": "training",
        "default_base": "Qwen/Qwen3-8B-Base",
        "adapter": "lora_adapter",
        "merged": "merged",
        "gguf": "dinner_q8.gguf",
    },
    "reaction": {
        "section": "reaction",
        "default_base": "Qwen/Qwen3-0.6B-Base",
        "adapter": "reaction_adapter",
        "merged": "reaction_merged",
        "gguf": "dinner_reaction_q8.gguf",
    },
}


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "persona"
    if target not in TARGETS:
        print(f"Unknown target '{target}'. Use one of: {', '.join(TARGETS)}")
        return
    spec = TARGETS[target]

    config = load_config()
    base_model = config.get(spec["section"], {}).get("base_model", spec["default_base"])

    adapter_path = os.path.join(ROOT_DIR, "data", "model", spec["adapter"])
    merged_path = os.path.join(ROOT_DIR, "data", "model", spec["merged"])
    os.makedirs(merged_path, exist_ok=True)

    print(f"[{target}] Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    print(f"[{target}] Loading LoRA adapter from {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    print(f"[{target}] Merging adapter into base model...")
    model = model.merge_and_unload()

    print(f"[{target}] Saving merged model to {merged_path}")
    model.save_pretrained(merged_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_path)

    print("\nMerge complete. Now convert to GGUF (Q8_0 directly, no separate quantize step):")
    print("  git clone https://github.com/ggerganov/llama.cpp")
    print("  pip install -r llama.cpp/requirements.txt")
    print(f"  python llama.cpp/convert_hf_to_gguf.py {merged_path} "
          f"--outfile data/model/{spec['gguf']} --outtype q8_0")


if __name__ == "__main__":
    main()
