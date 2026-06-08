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
BOT_MENTION_NAME = bot_config.get("mention_name", "dinner")
REPLY_ON_MENTION = bot_config.get("reply_on_mention", True)
REPLY_CHAIN_DECAY = bot_config.get("reply_chain_decay", 0.5)
REACTION_CHANCE = bot_config.get("reaction_chance", 0.001)
MAX_REACTION_TOKENS = bot_config.get("max_reaction_tokens", 16)
REACTION_TEMPERATURE = bot_config.get("reaction_temperature", 1.0)

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
    from llama_cpp import Llama, LlamaGrammar

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
intents.members = True
client = discord.Client(intents=intents)


def strip_thinking(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


def clean_mentions(msg):
    text = msg.content.strip()
    for user in msg.mentions:
        name = BOT_MENTION_NAME if BOT_MENTION_NAME and user.id == client.user.id else user.name
        text = re.sub(rf"<@!?{user.id}>", f"@{name}", text)
    return text


def build_prompt(context_messages, replied_to_id=None):
    lines = []
    for msg in context_messages:
        if msg.content.strip():
            prefix = "(replied to) " if replied_to_id and msg.id == replied_to_id else ""
            name = "you" if msg.author == client.user else msg.author.display_name
            content = clean_mentions(msg)
            lines.append(f"{prefix}{name}: {content}")
    return "\n".join(lines)


SYSTEM_PROMPT = "You are Dinner. Reply in character."


def generate_reply(context_text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + (" /no_think" if not USE_HF else "")},
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
        reply = strip_thinking(reply)
        reply = re.sub(r"https?://\S+", "", reply)
        return reply.strip()
    else:
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        reply = response["choices"][0]["message"]["content"]
        reply = strip_thinking(reply)
        reply = re.sub(r"https?://\S+", "", reply)
        return reply.strip()


REACTION_SYSTEM_PROMPT = "Pick one emoji reaction for the last message. Output ONLY the emoji."

UNICODE_EMOJIS = [
    "😂", "😭", "💀", "🔥", "❤️", "😍", "🥺", "😎", "🤣", "😤",
    "😡", "🤔", "😏", "🥰", "😳", "👍", "👎", "👀", "🙏", "💯",
    "🗿", "😈", "🤡", "💔", "✨", "🎉", "😐", "😑", "🤮", "🤢",
    "😴", "🤯", "🥲", "😮", "😱", "🫡", "🫠", "🤝", "✅", "❌",
    "⭐", "🐐", "💪", "😔", "😢", "🙄", "😬", "🤓", "💅", "👻",
    "🎶", "🤷", "👁️", "🧠", "😋", "🤨", "😊", "🫣", "🫵", "👍🏻",
]

EMOJI_GRAMMAR_BASE = 'unicode-emoji ::= ' + " | ".join(f'"{e}"' for e in UNICODE_EMOJIS)


def build_reaction_grammar(guild=None):
    if guild and guild.emojis:
        alternatives = " | ".join(f'":{e.name}:"' for e in guild.emojis)
        return f"root ::= {alternatives}"
    return "root ::= unicode-emoji\n" + EMOJI_GRAMMAR_BASE


def generate_reaction(context_text, guild=None):
    grammar = LlamaGrammar.from_string(build_reaction_grammar(guild))

    messages = [
        {"role": "system", "content": REACTION_SYSTEM_PROMPT + " /no_think"},
        {"role": "user", "content": context_text},
    ]

    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=MAX_REACTION_TOKENS,
        temperature=REACTION_TEMPERATURE,
        top_p=TOP_P,
        grammar=grammar,
    )
    result = response["choices"][0]["message"]["content"]
    result = strip_thinking(result)
    return result.strip()


def resolve_reaction_emoji(text, guild):
    custom_match = re.match(r"^:(\w+):$", text)
    if custom_match and guild:
        name = custom_match.group(1)
        for emoji in guild.emojis:
            if emoji.name.lower() == name.lower():
                return emoji
    return text


@client.event
async def on_ready():
    print(f"Bot online as {client.user}")
    print(f"Reply chance: {REPLY_CHANCE * 100}%")
    print(f"Reaction chance: {REACTION_CHANCE * 100}%")
    print(f"Allowed channels: {ALLOWED_CHANNELS}")


async def get_reply_chain_depth(message):
    depth = 0
    current = message
    while current.reference and current.reference.message_id:
        try:
            ref = current.reference.cached_message or await current.channel.fetch_message(current.reference.message_id)
        except (discord.NotFound, discord.HTTPException):
            break
        if ref.author == client.user:
            depth += 1
        current = ref
    return depth


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    in_allowed_channel = message.channel.id in ALLOWED_CHANNELS
    mentioned = REPLY_ON_MENTION and client.user in message.mentions
    replied_to_bot = (
        message.reference
        and message.reference.cached_message
        and message.reference.cached_message.author == client.user
    )
    forced = mentioned or replied_to_bot

    if not forced and not in_allowed_channel:
        return

    should_reply = False
    should_react = False

    if forced:
        should_reply = True
        if not mentioned:
            depth = await get_reply_chain_depth(message)
            if depth > 0 and random.random() > REPLY_CHAIN_DECAY ** depth:
                should_reply = False
    else:
        should_reply = random.random() <= REPLY_CHANCE

    if not USE_HF and in_allowed_channel:
        should_react = random.random() <= REACTION_CHANCE

    if not should_reply and not should_react:
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

    if should_react:
        try:
            print(f"[#{message.channel.name}] Generating reaction for: {message.content[:80]!r}")
            print(f"[#{message.channel.name}] Reaction context:\n{context_text}")
            reaction_text = generate_reaction(context_text, guild=message.guild)
            print(f"[#{message.channel.name}] Model output: {reaction_text!r}")
            if reaction_text:
                emoji = resolve_reaction_emoji(reaction_text, message.guild)
                print(f"[#{message.channel.name}] Resolved emoji: {emoji!r} (type: {type(emoji).__name__})")
                if emoji:
                    await message.add_reaction(emoji)
                    print(f"[#{message.channel.name}] Reacted with: {emoji}")
                else:
                    print(f"[#{message.channel.name}] Reaction unresolvable: {reaction_text!r}")
        except Exception as e:
            print(f"[#{message.channel.name}] Reaction failed: {e}")

    if should_reply:
        async with message.channel.typing():
            reply = generate_reply(context_text)

        if reply:
            guild = message.guild
            if guild:
                def resolve_mention(match):
                    name = match.group(1).lower()
                    for member in guild.members:
                        if member.name.lower() == name or (member.nick and member.nick.lower() == name):
                            return member.mention
                    return match.group(0)
                reply = re.sub(r"@(\w+)", resolve_mention, reply)

                def resolve_emoji(match):
                    name = match.group(1)
                    for emoji in guild.emojis:
                        if emoji.name.lower() == name.lower():
                            return str(emoji)
                    return match.group(0)
                reply = re.sub(r":(\w+):", resolve_emoji, reply)
            print(f"[#{message.channel.name}] Context:")
            print(context_text)
            print(f"  -> {reply}")
            await message.reply(reply, mention_author=False, allowed_mentions=discord.AllowedMentions(everyone=False))


def main():
    client.run(config["bot_token"])


if __name__ == "__main__":
    main()
