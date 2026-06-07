import os
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(ROOT_DIR, "config.yaml")) as f:
    config = yaml.safe_load(f)

merged_path = os.path.join(ROOT_DIR, "data", "model", "merged")

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(merged_path)
model = AutoModelForCausalLM.from_pretrained(
    merged_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()
print("Model loaded. Type a fake chat context and press Enter. Ctrl+C to quit.\n")

while True:
    context = input("Context (e.g. 'Friend: yo wanna play tonight'): ")
    messages = [
        {"role": "system", "content": "You are Dinner. Reply in character."},
        {"role": "user", "content": context},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            inputs,
            max_new_tokens=100,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    reply = tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True)
    print(f"Dinner: {reply}\n")
