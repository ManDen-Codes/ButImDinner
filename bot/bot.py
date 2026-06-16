import json
import os
import random
import re
import sys

import discord
import yaml

# Windows consoles default to cp1252; bot output (emoji, mentions) is UTF-8.
# Without this, printing an emoji-containing model output raises UnicodeEncodeError
# and drops the message before it can be dispatched.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)


def load_config():
    config_path = os.path.join(ROOT_DIR, "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


config = load_config()
bot_config = config.get("bot", {})

ENGAGEMENT_CHANCE = bot_config.get("engagement_chance", 0.05)
ALLOWED_CHANNELS = set(bot_config.get("allowed_channels", []))
MODEL_PATH = bot_config.get("model_path", "data/model/dinner.gguf")
CONTEXT_MESSAGES = bot_config.get("context_messages", 10)
MAX_TOKENS = bot_config.get("max_response_tokens", 256)
TEMPERATURE = bot_config.get("temperature", 0.8)
TOP_P = bot_config.get("top_p", 0.9)
BOT_MENTION_NAME = bot_config.get("mention_name", "dinner")
REPLY_ON_MENTION = bot_config.get("reply_on_mention", True)
REPLY_CHAIN_DECAY = bot_config.get("reply_chain_decay", 0.5)

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
        dtype=torch.bfloat16,
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

GIF_INDEX_PATH = os.path.join(ROOT_DIR, "data", "processed", "gif_index.json")
gif_index = {}
if os.path.exists(GIF_INDEX_PATH):
    with open(GIF_INDEX_PATH, encoding="utf-8") as f:
        gif_index = json.load(f)
    print(f"Loaded {len(gif_index)} GIFs from index")
else:
    print("No GIF index found — gif actions will be skipped")

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
client = discord.Client(intents=intents)

SYSTEM_PROMPT = (
    "You are Dinner in a Discord server. Given the conversation context, "
    "decide what to do. Output a JSON action: reply, react, gif, reply_react, "
    "or none. Reply in character."
)

UNICODE_EMOJIS = [
    "\U0001f602", "\U0001f62d", "\U0001f480", "\U0001f525", "❤️",
    "\U0001f60d", "\U0001f97a", "\U0001f60e", "\U0001f923", "\U0001f624",
    "\U0001f621", "\U0001f914", "\U0001f60f", "\U0001f970", "\U0001f633",
    "\U0001f44d", "\U0001f44e", "\U0001f440", "\U0001f64f", "\U0001f4af",
    "\U0001f5ff", "\U0001f608", "\U0001f921", "\U0001f494", "✨",
    "\U0001f389", "\U0001f610", "\U0001f611", "\U0001f92e", "\U0001f922",
    "\U0001f634", "\U0001f92f", "\U0001f972", "\U0001f62e", "\U0001f631",
    "\U0001fae1", "\U0001fae0", "\U0001f91d", "✅", "❌",
    "⭐", "\U0001f410", "\U0001f4aa", "\U0001f614", "\U0001f622",
    "\U0001f644", "\U0001f62c", "\U0001f913", "\U0001f485", "\U0001f47b",
    "\U0001f3b6", "\U0001f937", "\U0001f441️", "\U0001f9e0", "\U0001f60b",
    "\U0001f928", "\U0001f60a", "\U0001fae3", "\U0001fae5", "\U0001f44d\U0001f3fb",
]

# Grammar cache per guild
_grammar_cache = {}


def build_action_grammar(guild=None, allow_none=True):
    guild_id = guild.id if guild else None
    cache_key = (guild_id, allow_none)
    if cache_key in _grammar_cache:
        return _grammar_cache[cache_key]

    emoji_alts = " | ".join(f'"{e}"' for e in UNICODE_EMOJIS)
    if guild and guild.emojis:
        custom_alts = " | ".join(f'":{e.name}:"' for e in guild.emojis)
        emoji_rule = f"emoji ::= {emoji_alts} | {custom_alts}"
    else:
        emoji_rule = f"emoji ::= {emoji_alts}"

    # When the bot is directly addressed, drop the "none" option so it always acts.
    none_alt = " | action-none" if allow_none else ""

    grammar_str = rf"""root ::= action-reply | action-react{none_alt} | action-gif | action-reply-react

action-reply ::= "{{\"action\": \"reply\", \"text\": \"" text-content "\", \"mentions\": [" mentions-list "]}}"
action-react ::= "{{\"action\": \"react\", \"emoji\": \"" emoji "\"}}"
action-none ::= "{{\"action\": \"none\"}}"
action-gif ::= "{{\"action\": \"gif\", \"query\": \"" text-content "\"}}"
action-reply-react ::= "{{\"action\": \"reply_react\", \"text\": \"" text-content "\", \"emoji\": \"" emoji "\", \"mentions\": [" mentions-list "]}}"

mentions-list ::= "" | "\"" mention-name "\"" ("," " \"" mention-name "\"")*
mention-name ::= [a-zA-Z0-9_.]+

text-content ::= text-char+
text-char ::= [^"\\] | "\\" escape-char
escape-char ::= ["\\/bfnrt]

{emoji_rule}
"""
    grammar = LlamaGrammar.from_string(grammar_str)
    _grammar_cache[cache_key] = grammar
    return grammar


def strip_thinking(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


def clean_context(msg, guild):
    """Mirror process_v2.py clean_context_content so inference context matches
    training format exactly: strip role pings, resolve user/channel mentions to
    readable names, normalize custom emoji, replace URLs with [link]."""
    text = msg.content.strip()
    # Role mentions -> drop (training strips <@&id>; never feed raw role pings).
    text = re.sub(r"<@&\d+>", "", text)
    # User mentions -> @username (bot's own -> configured mention_name). Usernames are
    # stable; nicknames drift, so we match training which uses usernames.
    for user in msg.mentions:
        name = BOT_MENTION_NAME if BOT_MENTION_NAME and user.id == client.user.id else user.name
        text = re.sub(rf"<@!?{user.id}>", f"@{name}", text)
    text = re.sub(r"<@!?\d+>", "@unknown", text)
    # Channel mentions -> #name.
    def repl_channel(m):
        ch = guild.get_channel(int(m.group(1))) if guild else None
        return f"#{ch.name}" if ch else "#unknown"
    text = re.sub(r"<#(\d+)>", repl_channel, text)
    # Custom emoji -> :name:.
    text = re.sub(r"<a?:(\w+):\d+>", r":\1:", text)
    # URLs -> [link].
    text = re.sub(r"https?://\S+", "[link]", text)
    # Collapse doubled whitespace (matches training).
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def build_prompt(context_messages, guild=None, replied_to_id=None):
    lines = []
    for msg in context_messages:
        content = clean_context(msg, guild)
        # Attachment tagging mirrors process_v2.py build_context_lines.
        if msg.attachments:
            content = f"{content} [+attachment]" if content else "[shared media]"
        if content:
            prefix = "(replied to) " if replied_to_id and msg.id == replied_to_id else ""
            # Username (stable) for speaker labels, matching training; bot's own -> "you".
            name = "you" if msg.author == client.user else msg.author.name
            lines.append(f"{prefix}{name}: {content}")
    return "\n".join(lines)


def generate(context_text, guild=None, allow_none=True):
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
        raw = tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True)
        return strip_thinking(raw).strip()
    else:
        grammar = build_action_grammar(guild, allow_none)
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            grammar=grammar,
        )
        raw = response["choices"][0]["message"]["content"]
        return strip_thinking(raw).strip()


def parse_action(raw_output):
    json_match = re.search(r"\{[^{}]*\}", raw_output)
    if not json_match:
        return None
    try:
        action = json.loads(json_match.group())
    except json.JSONDecodeError:
        return None
    if action.get("action") not in ("reply", "react", "gif", "reply_react", "none"):
        return None
    return action


def resolve_output_mentions(text, mention_names, guild):
    if not guild:
        return text

    def resolve_mention(match):
        name = match.group(1).lower().rstrip(".")
        for member in guild.members:
            if member.name.lower() == name or (member.nick and member.nick.lower() == name):
                return member.mention
        return match.group(0)

    # Allow dots — new Discord usernames can contain them (e.g. dinner.lore).
    text = re.sub(r"@([a-zA-Z0-9_.]+)", resolve_mention, text)
    return text


def resolve_output_emojis(text, guild):
    if not guild:
        return text

    def resolve_emoji(match):
        name = match.group(1)
        for emoji in guild.emojis:
            if emoji.name.lower() == name.lower():
                return str(emoji)
        return match.group(0)

    return re.sub(r":(\w+):", resolve_emoji, text)


def resolve_reaction_emoji(text, guild):
    custom_match = re.match(r"^:(\w+):$", text)
    if custom_match and guild:
        name = custom_match.group(1)
        for emoji in guild.emojis:
            if emoji.name.lower() == name.lower():
                return emoji
    return text


def search_gif(query):
    if not gif_index:
        return None
    query_words = set(query.lower().split())
    scored = []
    for url, slug_words in gif_index.items():
        overlap = query_words & set(slug_words)
        if overlap:
            scored.append((len(overlap), url))
    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    top = [url for score, url in scored if score == scored[0][0]]
    return random.choice(top[:3])


async def dispatch_action(action, message):
    action_type = action["action"]
    guild = message.guild
    channel_tag = f"[#{message.channel.name}]"

    if action_type == "none":
        print(f"{channel_tag} Action: none")
        return

    if action_type in ("reply", "reply_react"):
        text = action.get("text", "")
        if text:
            mention_names = action.get("mentions", [])
            text = resolve_output_mentions(text, mention_names, guild)
            text = resolve_output_emojis(text, guild)
            text = re.sub(r"https?://\S+", "", text).strip()
            if text:
                print(f"{channel_tag} Action: {action_type} -> {text}")
                await message.reply(
                    text,
                    mention_author=False,
                    allowed_mentions=discord.AllowedMentions(everyone=False),
                )

    if action_type in ("react", "reply_react"):
        emoji_str = action.get("emoji", "")
        if emoji_str:
            emoji = resolve_reaction_emoji(emoji_str, guild)
            if emoji:
                print(f"{channel_tag} Action: react -> {emoji}")
                await message.add_reaction(emoji)

    if action_type == "gif":
        query = action.get("query", "")
        if query:
            gif_url = search_gif(query)
            if gif_url:
                print(f"{channel_tag} Action: gif ({query!r}) -> {gif_url}")
                await message.reply(
                    gif_url,
                    mention_author=False,
                    allowed_mentions=discord.AllowedMentions(everyone=False),
                )
            else:
                print(f"{channel_tag} No GIF match for {query!r}")


@client.event
async def on_ready():
    print(f"Bot online as {client.user}")
    print(f"Engagement chance: {ENGAGEMENT_CHANCE * 100}%")
    print(f"Allowed channels: {ALLOWED_CHANNELS}")
    print(f"GIF index: {len(gif_index)} GIFs loaded")


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

    should_engage = False
    if forced:
        should_engage = True
        if not mentioned:
            depth = await get_reply_chain_depth(message)
            if depth > 0 and random.random() > REPLY_CHAIN_DECAY ** depth:
                should_engage = False
    else:
        should_engage = random.random() <= ENGAGEMENT_CHANCE

    if not should_engage:
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

    context_text = build_prompt(history, guild=message.guild, replied_to_id=replied_to_id)
    if not context_text:
        return

    channel_tag = f"[#{message.channel.name}]"
    print(f"{channel_tag} Context:\n{context_text}")

    async with message.channel.typing():
        raw_output = generate(context_text, guild=message.guild, allow_none=not forced)

    print(f"{channel_tag} Raw output: {raw_output!r}")
    action = parse_action(raw_output)

    if action is None:
        if forced:
            cleaned = re.sub(r"https?://\S+", "", raw_output).strip()
            if cleaned:
                print(f"{channel_tag} Fallback plain reply: {cleaned}")
                await message.reply(
                    cleaned,
                    mention_author=False,
                    allowed_mentions=discord.AllowedMentions(everyone=False),
                )
        else:
            print(f"{channel_tag} Unparseable output, ignoring")
        return

    try:
        await dispatch_action(action, message)
    except discord.HTTPException as e:
        print(f"{channel_tag} Dispatch failed: {e}")
    except Exception as e:
        print(f"{channel_tag} Unexpected error: {e}")


@client.event
async def on_guild_emojis_update(guild, before, after):
    for key in [k for k in _grammar_cache if k[0] == guild.id]:
        _grammar_cache.pop(key, None)


def main():
    client.run(config["bot_token"])


if __name__ == "__main__":
    main()
