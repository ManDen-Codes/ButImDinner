import os
import random

import discord
import yaml
from llama_cpp import Llama

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

print(f"Loading model from {MODEL_PATH}...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_gpu_layers=-1,  # offload all layers to GPU
    verbose=False,
)
print("Model loaded!")

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


def build_prompt(context_messages, replied_to_id=None):
    """Format channel context the same way as training data."""
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
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        stop=["\n\n"],
    )
    return response["choices"][0]["message"]["content"].strip()


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
