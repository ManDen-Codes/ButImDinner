import os
import random
import re

import discord
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)


def load_config():
    config_path = os.path.join(ROOT_DIR, "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


config = load_config()
bot_config = config.get("bot", {})

REPLY_CHANCE = bot_config.get("reply_chance", 0.0001)
ALLOWED_CHANNELS = set(bot_config.get("allowed_channels", []))
MODEL_PATH = bot_config.get("model_path", "data/model/dinner.gguf")
CONTEXT_MESSAGES = bot_config.get("context_messages", 10)
MAX_TOKENS = bot_config.get("max_response_tokens", 256)
TEMPERATURE = bot_config.get("temperature", 0.8)
TOP_P = bot_config.get("top_p", 0.9)

if not os.path.isabs(MODEL_PATH):
    MODEL_PATH = os.path.join(ROOT_DIR, MODEL_PATH)

USE_HF = os.path.isdir(MODEL_PATH)

if USE_HF:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"Loading HF model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("Model loaded!")
else:
    from llama_cpp import Llama

    print(f"Loading GGUF model from {MODEL_PATH}...")
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_gpu_layers=-1,
        verbose=False,
    )
    print("Model loaded!")

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


def build_prompt(context_messages, replied_to_id=None):
    lines = []
    for msg in context_messages:
        if msg.content.strip():
            prefix = "(replied to) " if replied_to_id and msg.id == replied_to_id else ""
            lines.append(f"{prefix}{msg.author.display_name}: {msg.content.strip()}")
    return "\n".join(lines)


def generate_reply(context_text):
    messages = [
        {"role": "system", "content": "You are Dinner. Reply in character."},
        {"role": "user", "content": context_text},
    ]

    if USE_HF:
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
            enable_thinking=False,
        ).to(model.device)
        with torch.no_grad():
            output = model.generate(
                inputs,
                max_new_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        reply = tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True)
        reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL)
        reply = re.sub(r"https?://\S+", "", reply)
        return reply.strip()
    else:
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            stop=["\n\n"],
        )
        reply = response["choices"][0]["message"]["content"]
        reply = re.sub(r"https?://\S+", "", reply)
        return reply.strip()


@client.event
async def on_ready():
    print(f"Bot online as {client.user}")
    print(f"Reply chance: {REPLY_CHANCE * 100}%")
    print(f"Allowed channels: {ALLOWED_CHANNELS}")


@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if message.channel.id not in ALLOWED_CHANNELS:
        return
    if random.random() > REPLY_CHANCE:
        return

    history = []
    async for msg in message.channel.history(limit=CONTEXT_MESSAGES + 1):
        history.append(msg)
    history.reverse()

    replied_to_id = None
    if message.reference and message.reference.message_id:
        replied_to_id = message.reference.message_id
        if not any(m.id == replied_to_id for m in history):
            try:
                ref_msg = await message.channel.fetch_message(replied_to_id)
                history.insert(0, ref_msg)
            except discord.NotFound:
                pass

    context_text = build_prompt(history, replied_to_id=replied_to_id)
    if not context_text:
        return

    async with message.channel.typing():
        reply = generate_reply(context_text)

    if reply:
        await message.reply(reply, mention_author=False)


def main():
    client.run(config["bot_token"])


if __name__ == "__main__":
    main()
