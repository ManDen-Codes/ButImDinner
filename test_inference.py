import os
import re

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(SCRIPT_DIR, "config.yaml")) as f:
    config = yaml.safe_load(f)

bot_config = config.get("bot", {})
model_path = bot_config.get("model_path", "data/model/dinner.gguf")
if not os.path.isabs(model_path):
    model_path = os.path.join(SCRIPT_DIR, model_path)

USE_HF = os.path.isdir(model_path)


def strip_thinking(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


if USE_HF:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"Loading HF model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("Model loaded!\n")

    def generate(context_text, temperature=0.8):
        messages = [
            {"role": "system", "content": "You are Dinner. Reply in character."},
            {"role": "user", "content": context_text},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
            enable_thinking=False,
        ).to(model.device)
        with torch.no_grad():
            output = model.generate(
                inputs,
                max_new_tokens=256,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True)
        cleaned = strip_thinking(raw)
        cleaned = re.sub(r"https?://\S+", "", cleaned).strip()
        return raw, cleaned

else:
    from llama_cpp import Llama

    print(f"Loading GGUF model from {model_path}...")
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_gpu_layers=-1,
        verbose=False,
    )
    print("Model loaded!\n")

    def generate(context_text, temperature=0.8):
        messages = [
            {"role": "system", "content": "You are Dinner. Reply in character. /no_think"},
            {"role": "user", "content": context_text},
        ]
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=256,
            temperature=temperature,
            top_p=0.9,
        )
        raw = response["choices"][0]["message"]["content"]
        cleaned = strip_thinking(raw)
        cleaned = re.sub(r"https?://\S+", "", cleaned).strip()
        return raw, cleaned


print("Type a fake chat context (e.g. 'Bob: hey dinner whats up') or 'quit' to exit.\n")

while True:
    context = input("Context> ").strip()
    if context.lower() in ("quit", "exit", "q"):
        break
    if not context:
        continue

    raw, cleaned = generate(context)
    print(f"  Raw:     {raw}")
    print(f"  Cleaned: {cleaned}")
    if not cleaned:
        print("  (empty after stripping -- model only produced thinking tokens)")
    print()
