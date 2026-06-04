import os

import yaml
from unsloth import FastLanguageModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)


def load_config():
    config_path = os.path.join(ROOT_DIR, "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    tc = config.get("training", {})
    max_seq_length = tc.get("max_seq_length", 2048)

    adapter_path = os.path.join(ROOT_DIR, "data", "model", "lora_adapter")
    output_dir = os.path.join(ROOT_DIR, "data", "model")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model with LoRA adapter from {adapter_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )

    output_path = os.path.join(output_dir, "dinner.gguf")
    print(f"Exporting to GGUF: {output_path}")
    model.save_pretrained_gguf(
        output_dir,
        tokenizer,
        quantization_method="q4_k_m",
    )

    # Unsloth saves as <model_name>-unsloth-Q4_K_M.gguf, rename it
    for f in os.listdir(output_dir):
        if f.endswith(".gguf") and f != "dinner.gguf":
            src = os.path.join(output_dir, f)
            os.rename(src, output_path)
            break

    print(f"Done! Model saved to {output_path}")


if __name__ == "__main__":
    main()
