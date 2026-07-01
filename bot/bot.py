import json
import os
import random
import re
import sys

import discord
import yaml

# Windows consoles default to cp1252; bot output (emoji, mentions) is UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)


def load_config():
    with open(os.path.join(ROOT_DIR, "config.yaml")) as f:
        return yaml.safe_load(f)


config = load_config()
bot_config = config.get("bot", {})
proc_config = config.get("processing", {})

ENGAGEMENT_CHANCE = bot_config.get("engagement_chance", 0.05)
ALLOWED_CHANNELS = set(bot_config.get("allowed_channels", []))
MODEL_PATH = bot_config.get("model_path", "data/model/dinner_q8.gguf")
REACTION_MODEL_PATH = bot_config.get("reaction_model_path", "")
CONTEXT_MESSAGES = bot_config.get("context_messages", 12)
MAX_TOKENS = bot_config.get("max_response_tokens", 256)
TEMPERATURE = bot_config.get("temperature", 0.8)
TOP_P = bot_config.get("top_p", 0.9)
REPLY_ON_MENTION = bot_config.get("reply_on_mention", True)
REPLY_CHAIN_DECAY = bot_config.get("reply_chain_decay", 0.5)
# Stable username Dinner trained under — MUST match process_v2 primary_name (the
# username for dinner_user_id). Used as his speaker label and self-mention name.
PERSONA_NAME = bot_config.get("persona_name", "dinnerlore")
GIF_ENABLED = bot_config.get("gif_enabled", True)
REACT_CHANCE = bot_config.get("react_chance", 0.05)
MAX_SEGMENTS = bot_config.get("max_reply_segments", 4)
MERGE_WINDOW = proc_config.get("merge_window_minutes", 5) * 60
# Context caps — MUST match process_v2 (context_max_turns / context_max_chars) so the
# inference transcript is shaped exactly like the training prompts.
CONTEXT_MAX_TURNS = proc_config.get("context_max_turns", 12)
CONTEXT_MAX_CHARS = proc_config.get("context_max_chars", 1500)
# Fetch enough raw messages to fill the turn cap even after same-author merging.
FETCH_LIMIT = max(CONTEXT_MESSAGES, CONTEXT_MAX_TURNS * 2)

# Cue appended to the reaction model's prompt — MUST match process_v2.REACT_CUE.
REACT_CUE = "\n[reaction]:"

TENOR_RE = re.compile(r"https://tenor\.com/view/(.+?)-(\d+)")
GIF_TOKEN_RE = re.compile(r"\[gif:\s*(.*?)\]")
URL_RE = re.compile(r"https?://\S+")


def resolve_path(p):
    if p and not os.path.isabs(p):
        return os.path.join(ROOT_DIR, p)
    return p


def load_model(path, n_ctx=2048):
    """Auto-detect backend: directory -> HF transformers, .gguf file -> llama.cpp.
    Returns a dict describing the loaded model, or None if path is empty/missing."""
    path = resolve_path(path)
    if not path or not os.path.exists(path):
        return None
    if os.path.isdir(path):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        print(f"Loading HF model from {path}...")
        tok = AutoTokenizer.from_pretrained(path)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        mdl = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="auto")
        mdl.eval()
        return {"kind": "hf", "model": mdl, "tokenizer": tok, "torch": torch}
    else:
        from llama_cpp import Llama, LlamaGrammar
        print(f"Loading GGUF model from {path}...")
        llm = Llama(model_path=path, n_ctx=n_ctx, n_gpu_layers=-1, verbose=False)
        return {"kind": "gguf", "llm": llm, "LlamaGrammar": LlamaGrammar}


print("Loading persona model...")
persona = load_model(MODEL_PATH)
if persona is None:
    raise SystemExit(f"Persona model not found at {resolve_path(MODEL_PATH)}")
print("Persona model loaded!")

reaction = load_model(REACTION_MODEL_PATH)
print("Reaction model loaded!" if reaction else
      "No reaction model — using emoji-frequency heuristic.")

# --- gif index + reaction emoji frequencies ---
def load_json(rel):
    p = os.path.join(ROOT_DIR, rel)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


gif_index = load_json("data/processed/gif_index.json")
print(f"Loaded {len(gif_index)} GIFs from index" if gif_index else "No GIF index found")

REACTION_FREQ = load_json("data/processed/reaction_emoji_freq.json")
REACTION_VOCAB = list(REACTION_FREQ.keys())
print(f"Loaded {len(REACTION_VOCAB)} reaction emoji")

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
client = discord.Client(intents=intents)

_reaction_grammar = None


def reaction_grammar():
    """GBNF constraining the reaction model to Dinner's emoji vocab or 'none'."""
    global _reaction_grammar
    if reaction is None or reaction["kind"] != "gguf":
        return None
    if _reaction_grammar is None:
        def lit(s):
            return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
        alts = " | ".join(lit(e) for e in REACTION_VOCAB) or '"none"'
        grammar_str = f'root ::= " "? ( {alts} | "none" )\n'
        _reaction_grammar = reaction["LlamaGrammar"].from_string(grammar_str)
    return _reaction_grammar


# ---------------------------------------------------------------------------
# Context rendering — MUST mirror process_v2.render_message (is_target=False)
# so inference context matches the training transcript format exactly.
# ---------------------------------------------------------------------------
def render_ctx_msg(msg, guild):
    text = msg.content
    text = TENOR_RE.sub(lambda m: "[gif: " + " ".join(m.group(1).lower().split("-")) + "]", text)
    text = re.sub(r"<@&\d+>", "", text)                       # role mentions: drop
    for u in msg.mentions:
        name = PERSONA_NAME if (client.user and u.id == client.user.id) else u.name
        text = re.sub(rf"<@!?{u.id}>", f"@{name}", text)
    text = re.sub(r"<@!?\d+>", "@unknown", text)
    def repl_channel(m):
        ch = guild.get_channel(int(m.group(1))) if guild else None
        return f"#{ch.name}" if ch else "#unknown"
    text = re.sub(r"<#(\d+)>", repl_channel, text)
    text = re.sub(r"<a?:(\w+):\d+>", r":\1:", text)           # custom emoji -> :name:
    text = URL_RE.sub("[link]", text)                          # non-gif URLs
    text = re.sub(r"\s{2,}", " ", text).strip()
    if msg.attachments:
        text = (text + " [image]").strip() if text else "[image]"
    return text or None


def speaker_name(msg):
    return PERSONA_NAME if (client.user and msg.author.id == client.user.id) else msg.author.name


def build_transcript(history, guild):
    """Chronological messages -> transcript with consecutive same-author merge."""
    turns = []
    for msg in history:
        rendered = render_ctx_msg(msg, guild)
        if rendered is None:
            continue
        aid = msg.author.id
        ts = msg.created_at.timestamp()
        if turns and turns[-1]["aid"] == aid and ts - turns[-1]["ts"] <= MERGE_WINDOW:
            turns[-1]["parts"].append(rendered)
            turns[-1]["ts"] = ts
        else:
            turns.append({"aid": aid, "name": speaker_name(msg), "parts": [rendered], "ts": ts})
    # Cap to the last N turns, then trim to the char budget from the most recent end —
    # identical to process_v2.build_context so inference matches training.
    turns = turns[-CONTEXT_MAX_TURNS:]
    lines = [f"{t['name']}: " + "\n".join(t["parts"]) for t in turns]
    out, kept, total = [], [], 0
    for line, t in zip(reversed(lines), reversed(turns)):
        if out and total + len(line) > CONTEXT_MAX_CHARS:
            break
        out.append(line)
        kept.append(t)
        total += len(line)
    out.reverse()
    names = {t["name"] for t in kept} | {PERSONA_NAME}
    return "\n".join(out), names


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def _truncate_at_speaker(text, known_names):
    cut = len(text)
    for n in known_names:
        idx = text.find(f"\n{n}:")
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut]


def generate_persona(prompt, known_names):
    stops = [f"\n{n}:" for n in known_names]
    if persona["kind"] == "hf":
        torch = persona["torch"]
        tok = persona["tokenizer"]
        inputs = tok(prompt, return_tensors="pt").to(persona["model"].device)
        with torch.no_grad():
            out = persona["model"].generate(
                **inputs, max_new_tokens=MAX_TOKENS, temperature=TEMPERATURE,
                top_p=TOP_P, do_sample=True, pad_token_id=tok.eos_token_id,
            )
        raw = tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    else:
        resp = persona["llm"](prompt, max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
                              top_p=TOP_P, stop=stops[:8])
        raw = resp["choices"][0]["text"]
    raw = raw.strip()
    if raw.lower().startswith(PERSONA_NAME.lower() + ":"):
        raw = raw[len(PERSONA_NAME) + 1:].strip()
    return _truncate_at_speaker(raw, known_names).strip()


def generate_reaction(prompt):
    if reaction["kind"] == "hf":
        torch = reaction["torch"]
        tok = reaction["tokenizer"]
        inputs = tok(prompt, return_tensors="pt").to(reaction["model"].device)
        with torch.no_grad():
            out = reaction["model"].generate(
                **inputs, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id,
            )
        return tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    resp = reaction["llm"](prompt, max_tokens=12, temperature=0.3,
                           grammar=reaction_grammar(), stop=["\n"])
    return resp["choices"][0]["text"].strip()


def decide_reaction(transcript):
    """Return an emoji string (':name:' or unicode) or None."""
    if reaction is not None:
        out = generate_reaction(transcript + REACT_CUE)
        low = out.strip().lower()
        if not low or low.startswith("none"):
            return None
        for e in REACTION_VOCAB:
            if e in out:
                return e
        m = re.search(r":(\w+):", out)
        if m:
            return m.group(0)
        token = out.split()[0] if out.split() else None
        return token
    # phase-1 heuristic: weighted draw from Dinner's reaction history
    if not REACTION_FREQ:
        return None
    return random.choices(REACTION_VOCAB, weights=list(REACTION_FREQ.values()))[0]


# ---------------------------------------------------------------------------
# Output handling
# ---------------------------------------------------------------------------
def resolve_output_mentions(text, guild):
    if not guild:
        return text
    def resolve(match):
        name = match.group(1).lower().rstrip(".")
        for member in guild.members:
            if member.name.lower() == name or (member.nick and member.nick.lower() == name):
                return member.mention
        return match.group(0)
    return re.sub(r"@([a-zA-Z0-9_.]+)", resolve, text)


def resolve_output_emojis(text, guild):
    if not guild:
        return text
    def resolve(match):
        name = match.group(1)
        for emoji in guild.emojis:
            if emoji.name.lower() == name.lower():
                return str(emoji)
        return match.group(0)
    return re.sub(r":(\w+):", resolve, text)


def resolve_reaction_emoji(text, guild):
    m = re.match(r"^:(\w+):$", text)
    if m and guild:
        for emoji in guild.emojis:
            if emoji.name.lower() == m.group(1).lower():
                return emoji
    return text


def search_gif(query):
    if not gif_index:
        return None
    query_words = set(query.lower().split())
    scored = [(len(query_words & set(words)), url)
              for url, words in gif_index.items() if query_words & set(words)]
    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    top = [url for score, url in scored if score == scored[0][0]]
    return random.choice(top[:3])


def split_segments(raw, guild):
    """Split a (possibly multi-line, gif-containing) completion into ordered
    send segments: ('text', str) and ('gif', url)."""
    segments = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        gifs = GIF_TOKEN_RE.findall(line)
        text = GIF_TOKEN_RE.sub("", line).strip()
        if text:
            text = resolve_output_mentions(text, guild)
            text = resolve_output_emojis(text, guild)
            text = URL_RE.sub("", text).strip()
            if text:
                segments.append(("text", text))
        if GIF_ENABLED:
            for q in gifs:
                url = search_gif(q)
                if url:
                    segments.append(("gif", url))
    return segments[:MAX_SEGMENTS]


async def send_segments(segments, message):
    allowed = discord.AllowedMentions(everyone=False)
    first = True
    for kind, payload in segments:
        if first:
            await message.reply(payload, mention_author=False, allowed_mentions=allowed)
            first = False
        else:
            await message.channel.send(payload, allowed_mentions=allowed)
        print(f"  -> {kind}: {payload}")


# ---------------------------------------------------------------------------
# Discord events
# ---------------------------------------------------------------------------
@client.event
async def on_ready():
    print(f"Bot online as {client.user}")
    print(f"Persona name: {PERSONA_NAME} | engagement: {ENGAGEMENT_CHANCE:.0%} | "
          f"react: {REACT_CHANCE:.0%} | gifs: {GIF_ENABLED}")
    print(f"Allowed channels: {ALLOWED_CHANNELS}")


async def get_reply_chain_depth(message):
    depth, current = 0, message
    while current.reference and current.reference.message_id:
        try:
            ref = current.reference.cached_message or \
                await current.channel.fetch_message(current.reference.message_id)
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

    in_allowed = message.channel.id in ALLOWED_CHANNELS
    mentioned = REPLY_ON_MENTION and client.user in message.mentions
    replied_to_bot = (
        message.reference and message.reference.cached_message
        and message.reference.cached_message.author == client.user
    )
    forced = mentioned or replied_to_bot

    if not forced and not in_allowed:
        return

    # --- decide whether to reply ---
    should_engage = False
    if forced:
        should_engage = True
        if not mentioned:
            depth = await get_reply_chain_depth(message)
            if depth > 0 and random.random() > REPLY_CHAIN_DECAY ** depth:
                should_engage = False
    else:
        should_engage = random.random() <= ENGAGEMENT_CHANCE

    # --- decide whether to react (independent of replying) ---
    do_react = should_engage or (in_allowed and random.random() < REACT_CHANCE)

    if not should_engage and not do_react:
        return

    history = []
    async for msg in message.channel.history(limit=FETCH_LIMIT + 1):
        history.append(msg)
    history.reverse()

    transcript, known_names = build_transcript(history, message.guild)
    if not transcript:
        return

    channel_tag = f"[#{message.channel.name}]"

    if should_engage:
        prompt = f"{transcript}\n{PERSONA_NAME}:"
        async with message.channel.typing():
            raw = generate_persona(prompt, known_names)
        print(f"{channel_tag} Raw: {raw!r}")
        segments = split_segments(raw, message.guild)
        if segments:
            try:
                await send_segments(segments, message)
            except discord.HTTPException as e:
                print(f"{channel_tag} Send failed: {e}")

    if do_react and message.guild:
        try:
            emoji_str = decide_reaction(transcript)
            if emoji_str:
                emoji = resolve_reaction_emoji(emoji_str, message.guild)
                await message.add_reaction(emoji)
                print(f"{channel_tag} React: {emoji_str}")
        except (discord.HTTPException, discord.InvalidArgument) as e:
            print(f"{channel_tag} React failed: {e}")
        except Exception as e:
            print(f"{channel_tag} React error: {e}")


def main():
    client.run(config["bot_token"])


if __name__ == "__main__":
    main()
