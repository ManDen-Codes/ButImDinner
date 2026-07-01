"""v2 (official) processor — plain-text persona clone, no action JSON.

Produces two datasets:
  1. Persona model: context transcript -> Dinner's next message (plain text).
     - GIFs are an inline [gif: <slug words>] token, symmetric on input context
       AND output target (his Tenor posts become targets; others' Tenor posts in
       context render the same way instead of [link]).
     - Mentions stay inline as @username; emoji stay inline as :name:/unicode.
     - Conversation grouping (split on >gap), consecutive same-sender merge (<window),
       dominant-speaker balancing, recency weighting.
  2. Reaction model: context transcript -> a single reaction emoji or "none"
     (sparse + downsampled so it doesn't collapse to always-"none").

Also writes gif_index.json (unchanged contract) and reaction_emoji_freq.json
(for the bot's phase-1 reaction heuristic).
"""

import glob
import json
import os
import random
import re
from collections import defaultdict
from datetime import datetime, timezone

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
USER_INDEX_PATH = os.path.join(PROCESSED_DIR, "user_index.json")

URL_RE = re.compile(r"https?://\S+")
TENOR_RE = re.compile(r"https://tenor\.com/view/(.+?)-(\d+)")
MENTION_RE = re.compile(r"<@!?(\d+)>")
CUSTOM_EMOJI_RE = re.compile(r"<a?:(\w+):\d+>")

# Cue appended after the context for the reaction model. Must match bot.py.
REACT_CUE = "\n[reaction]:"


def normalize_emoji(emoji):
    """Custom emoji come from Discord as '<:name:id>'; the bot uses ':name:'.
    Unicode emoji pass through."""
    m = CUSTOM_EMOJI_RE.fullmatch(emoji)
    return f":{m.group(1)}:" if m else emoji


def load_config():
    with open(os.path.join(ROOT_DIR, "config.yaml")) as f:
        return yaml.safe_load(f)


def get_dinner_ids(config):
    ids = {config["dinner_user_id"]}
    ids.update(config.get("dinner_alt_ids") or [])
    return ids


def load_all_messages():
    channels = {}
    for path in glob.glob(os.path.join(RAW_DIR, "*.json")):
        with open(path, encoding="utf-8") as f:
            messages = json.load(f)
        if messages:
            channels[messages[0]["channel_name"]] = messages
    return channels


def load_user_index():
    if not os.path.exists(USER_INDEX_PATH):
        print("  WARNING: user_index.json not found — falling back to scraped display "
              "names. Run scraper/fetch_usernames.py for stable usernames.")
        return {}
    with open(USER_INDEX_PATH, encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f).items()}


def build_user_map(channels, username_index=None):
    user_map = {}
    for messages in channels.values():
        for msg in messages:
            user_map[msg["author_id"]] = msg["author_name"]
    if username_index:
        user_map.update(username_index)
    return user_map


def build_channel_map(channels):
    ch_map = {}
    for messages in channels.values():
        if messages:
            ch_map[messages[0]["channel_id"]] = messages[0]["channel_name"]
    return ch_map


def clean_content(content, user_map, channel_map):
    """Resolve mentions/channels to readable names, normalize custom emoji,
    collapse whitespace. Does NOT touch URLs (handled by the caller)."""
    def replace_user_mention(match):
        return f"@{user_map.get(int(match.group(1)), 'unknown')}"

    def replace_channel_mention(match):
        return f"#{channel_map.get(int(match.group(1)), 'unknown')}"

    content = re.sub(r"<@&\d+>", "", content)          # role mentions: never teach pinging roles
    content = re.sub(r"<@!?(\d+)>", replace_user_mention, content)
    content = re.sub(r"<#(\d+)>", replace_channel_mention, content)
    content = re.sub(r"<a?:(\w+):\d+>", r":\1:", content)
    return re.sub(r"\s{2,}", " ", content).strip()


def tenor_to_token(content):
    """Replace any Tenor URL inline with [gif: slug words]."""
    def sub(m):
        return "[gif: " + " ".join(m.group(1).lower().split("-")) + "]"
    return TENOR_RE.sub(sub, content)


def render_message(msg, user_map, channel_map, is_target):
    """Render one raw message to transcript text, or None if it carries no
    reproducible content.

    Targets (Dinner's own messages) drop anything with a non-Tenor URL or an
    attachment-only body (can't reproduce a link/image). Context keeps a [link]
    / [image] placeholder so the model still sees that something was shared.
    """
    content = tenor_to_token(msg["content"])
    content = clean_content(content, user_map, channel_map)
    has_attachment = bool(msg.get("attachments"))

    if URL_RE.search(content):                 # leftover real URL (not a gif token)
        if is_target:
            return None
        content = URL_RE.sub("[link]", content)

    if has_attachment:
        if is_target:
            return None if not content else content
        content = (content + " [image]").strip() if content else "[image]"

    content = content.strip()
    return content or None


def parse_ts(ts_str):
    ts = datetime.fromisoformat(ts_str)
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def split_into_blocks(messages, gap_seconds):
    """Split a channel's chronological messages into conversation blocks wherever
    the gap between consecutive messages exceeds gap_seconds."""
    blocks, current, last_ts = [], [], None
    for msg in messages:
        ts = parse_ts(msg["timestamp"])
        if last_ts is not None and (ts - last_ts).total_seconds() > gap_seconds:
            if current:
                blocks.append(current)
            current = []
        current.append(msg)
        last_ts = ts
    if current:
        blocks.append(current)
    return blocks


def block_to_turns(block, dinner_ids, primary_name, user_map, channel_map, merge_seconds):
    """Merge consecutive same-author messages within merge_seconds into one turn.
    Returns turns: {author_id, name, is_dinner, text, ts, msg_ids}."""
    # group consecutive same-author runs within the merge window
    runs = []
    for msg in block:
        ts = parse_ts(msg["timestamp"])
        if runs:
            prev_author, prev_ts, _ = runs[-1]
            if msg["author_id"] == prev_author and (ts - prev_ts).total_seconds() <= merge_seconds:
                runs[-1][1] = ts
                runs[-1][2].append(msg)
                continue
        runs.append([msg["author_id"], ts, [msg]])

    turns = []
    for author_id, last_ts, msgs in runs:
        is_dinner = author_id in dinner_ids
        parts = [render_message(m, user_map, channel_map, is_target=is_dinner) for m in msgs]
        parts = [p for p in parts if p]
        if not parts:
            continue
        name = primary_name if is_dinner else user_map.get(author_id, msgs[0]["author_name"])
        turns.append({
            "author_id": author_id,
            "name": name,
            "is_dinner": is_dinner,
            "text": "\n".join(parts),
            "ts": last_ts.isoformat(),
            "msg_ids": [m["id"] for m in msgs],
        })
    return turns


def build_context(turns, end_exclusive, max_turns, max_chars):
    """Transcript of up to max_turns turns ending before end_exclusive, trimmed
    to max_chars from the most recent end."""
    sel = turns[max(0, end_exclusive - max_turns):end_exclusive]
    lines = [f"{t['name']}: {t['text']}" for t in sel]
    out, total = [], 0
    for line in reversed(lines):
        if out and total + len(line) > max_chars:
            break
        out.append(line)
        total += len(line)
    return "\n".join(reversed(out))


def last_interlocutor(turns, end_exclusive):
    """Most recent non-Dinner speaker in the context (for dominant-speaker balancing)."""
    for t in reversed(turns[:end_exclusive]):
        if not t["is_dinner"]:
            return t["author_id"]
    return None


def recency_multiplier(ts_str, now, weights):
    months_old = (now - parse_ts(ts_str)).days / 30.44
    for bucket in weights:
        if months_old <= bucket["months"]:
            return bucket["weight"]
    return weights[-1]["weight"]


def build_gif_index(channels, dinner_ids):
    index = {}
    for messages in channels.values():
        for msg in messages:
            if msg["author_id"] not in dinner_ids:
                continue
            m = TENOR_RE.search(msg["content"])
            if m:
                index[m.group(0)] = m.group(1).lower().split("-")
    return index


def dinner_reaction_map(messages, dinner_ids):
    """msg_id -> first emoji Dinner reacted with (normalized)."""
    out = {}
    for msg in messages:
        for r in msg.get("reactions", []):
            if any(d in r.get("user_ids", []) for d in dinner_ids):
                out.setdefault(msg["id"], normalize_emoji(r["emoji"]))
    return out


def balance_dominant(pairs, cap_ratio):
    """Downsample any single interlocutor that exceeds cap_ratio of all pairs,
    so the clone doesn't overfit to one relationship (Watson Chua's failure)."""
    total = len(pairs)
    if total == 0 or cap_ratio >= 1:
        return pairs
    max_allowed = max(1, int(total * cap_ratio))
    by = defaultdict(list)
    for p in pairs:
        by[p["interlocutor"]].append(p)
    random.seed(42)
    out, capped = [], 0
    for lst in by.values():
        if len(lst) > max_allowed:
            capped += len(lst) - max_allowed
            out.extend(random.sample(lst, max_allowed))
        else:
            out.extend(lst)
    random.shuffle(out)
    return out, capped


def build_datasets(channels, config, now, username_index=None):
    dinner_ids = get_dinner_ids(config)
    pc = config.get("processing", {})
    gap_seconds = pc.get("conversation_gap_minutes", 60) * 60
    merge_seconds = pc.get("merge_window_minutes", 5) * 60
    max_turns = pc.get("context_max_turns", 12)
    max_chars = pc.get("context_max_chars", 1500)
    cap_ratio = pc.get("dominant_speaker_cap_ratio", 0.2)
    min_reply_chars = pc.get("min_reply_chars", 2)
    gif_oversample = pc.get("gif_oversample", 1)   # 1 = natural rate; raise to make gifs fire more
    react_none_ratio = pc.get("reaction_none_ratio", 1.0)
    react_min_count = pc.get("reaction_min_emoji_count", 5)
    react_top_k = pc.get("reaction_top_k_emoji", 25)

    weights = config.get("recency_weights", [{"months": 9999, "weight": 1}])

    user_map = build_user_map(channels, username_index)
    primary_name = user_map.get(config["dinner_user_id"], "dinnerlore")
    for did in dinner_ids:
        user_map[did] = primary_name
    channel_map = build_channel_map(channels)

    base_persona = []            # one entry per Dinner turn (pre recency/balance)
    react_positives = []         # (context, emoji, ts)
    react_none_refs = []         # (channel, block_idx, turn_idx) -> rebuilt after sampling
    emoji_counts = defaultdict(int)
    blocks_by_channel = {}

    for channel_name, messages in channels.items():
        messages = sorted(messages, key=lambda m: parse_ts(m["timestamp"]))
        reactions = dinner_reaction_map(messages, dinner_ids)
        for emo in reactions.values():
            emoji_counts[emo] += 1

        blocks = [
            block_to_turns(b, dinner_ids, primary_name, user_map, channel_map, merge_seconds)
            for b in split_into_blocks(messages, gap_seconds)
        ]
        blocks_by_channel[channel_name] = blocks
        dinner_in_channel = any(t["is_dinner"] for blk in blocks for t in blk)

        for b_idx, turns in enumerate(blocks):
            block_msg_react = {}  # turn_idx -> emoji for turns Dinner reacted to
            for t_idx, turn in enumerate(turns):
                # --- persona pairs: Dinner's turns with preceding context ---
                if turn["is_dinner"]:
                    context = build_context(turns, t_idx, max_turns, max_chars)
                    if not context:
                        continue
                    if "[gif:" not in turn["text"] and len(turn["text"]) < min_reply_chars:
                        continue
                    base_persona.append({
                        "prompt": f"{context}\n{primary_name}:",
                        "completion": " " + turn["text"],
                        "interlocutor": last_interlocutor(turns, t_idx),
                        "ts": turn["ts"],
                    })
                else:
                    # reaction label for this (non-Dinner) turn
                    emo = next((reactions[mid] for mid in turn["msg_ids"] if mid in reactions), None)
                    if emo:
                        block_msg_react[t_idx] = emo

            # --- reaction dataset (only where Dinner participates in this channel) ---
            if not dinner_in_channel:
                continue
            for t_idx, turn in enumerate(turns):
                if turn["is_dinner"]:
                    continue
                context = build_context(turns, t_idx + 1, max_turns, max_chars)
                if not context:
                    continue
                if t_idx in block_msg_react:
                    react_positives.append((context, block_msg_react[t_idx], turn["ts"]))
                else:
                    react_none_refs.append((channel_name, b_idx, t_idx))

    # ---- persona: balance dominant speaker, then recency-weight, then split ----
    base_persona, capped = balance_dominant(base_persona, cap_ratio)
    persona_pairs = []
    for p in base_persona:
        mult = recency_multiplier(p["ts"], now, weights)
        if gif_oversample > 1 and "[gif:" in p["completion"]:
            mult *= gif_oversample
        for _ in range(mult):
            persona_pairs.append({"prompt": p["prompt"], "completion": p["completion"]})
    random.seed(42)
    random.shuffle(persona_pairs)

    # ---- reactions: build emoji vocab, filter, downsample none, split ----
    vocab = {e for e, c in emoji_counts.items() if c >= react_min_count}
    if react_top_k:
        vocab = set(sorted(vocab, key=lambda e: -emoji_counts[e])[:react_top_k])
    react_positives = [(c, e) for (c, e, _ts) in react_positives if e in vocab]

    target_none = min(len(react_none_refs), int(len(react_positives) * react_none_ratio))
    random.seed(42)
    sampled_none = random.sample(react_none_refs, target_none) if target_none else []
    react_examples = [{"prompt": c + REACT_CUE, "completion": " " + e} for c, e in react_positives]
    for channel_name, b_idx, t_idx in sampled_none:
        turns = blocks_by_channel[channel_name][b_idx]
        context = build_context(turns, t_idx + 1, max_turns, max_chars)
        if context:
            react_examples.append({"prompt": context + REACT_CUE, "completion": " none"})
    random.seed(42)
    random.shuffle(react_examples)

    reaction_freq = dict(sorted(emoji_counts.items(), key=lambda x: -x[1])[:react_top_k or 25])

    stats = {
        "persona_base": len(base_persona),
        "persona_weighted": len(persona_pairs),
        "dominant_capped": capped,
        "react_positive": len(react_positives),
        "react_none": len(react_examples) - len(react_positives),
        "emoji_vocab": len(vocab),
    }
    return persona_pairs, react_examples, reaction_freq, stats


def split_jsonl(rows, train_ratio, key=None):
    random.seed(42)
    if key is None:
        rows = list(rows)
        random.shuffle(rows)
        cut = int(len(rows) * train_ratio)
        return rows[:cut], rows[cut:]
    buckets = defaultdict(list)
    for r in rows:
        buckets[key(r)].append(r)
    train, val = [], []
    for lst in buckets.values():
        random.shuffle(lst)
        cut = int(len(lst) * train_ratio)
        train.extend(lst[:cut])
        val.extend(lst[cut:])
    random.shuffle(train)
    random.shuffle(val)
    return train, val


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    config = load_config()
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    now = datetime.now(timezone.utc)
    train_ratio = config.get("processing", {}).get("train_ratio", 0.9)

    print("Loading raw messages...")
    channels = load_all_messages()

    cutoff = config.get("scraper", {}).get("cutoff_date")
    if cutoff:
        cutoff_dt = datetime.fromisoformat(cutoff).replace(tzinfo=timezone.utc)
        for ch in list(channels.keys()):
            channels[ch] = [m for m in channels[ch] if parse_ts(m["timestamp"]) <= cutoff_dt]
        print(f"Cutoff date applied: {cutoff}")

    total = sum(len(m) for m in channels.values())
    print(f"Loaded {total} messages from {len(channels)} channels")

    print("Building GIF index...")
    gif_index = build_gif_index(channels, get_dinner_ids(config))
    with open(os.path.join(PROCESSED_DIR, "gif_index.json"), "w", encoding="utf-8") as f:
        json.dump(gif_index, f, ensure_ascii=False, indent=2)
    print(f"  {len(gif_index)} unique GIFs")

    print("Loading username index...")
    username_index = load_user_index()
    print(f"  {len(username_index)} usernames loaded")

    print("Building datasets...")
    random.seed(42)
    persona, reactions, reaction_freq, stats = build_datasets(channels, config, now, username_index)

    if not persona:
        print("No persona pairs found. Check dinner_user_id.")
        return

    p_train, p_val = split_jsonl(persona, train_ratio)
    r_train, r_val = split_jsonl(reactions, train_ratio, key=lambda r: r["completion"])

    write_jsonl(os.path.join(PROCESSED_DIR, "train.jsonl"), p_train)
    write_jsonl(os.path.join(PROCESSED_DIR, "val.jsonl"), p_val)
    write_jsonl(os.path.join(PROCESSED_DIR, "reactions_train.jsonl"), r_train)
    write_jsonl(os.path.join(PROCESSED_DIR, "reactions_val.jsonl"), r_val)
    with open(os.path.join(PROCESSED_DIR, "reaction_emoji_freq.json"), "w", encoding="utf-8") as f:
        json.dump(reaction_freq, f, ensure_ascii=False, indent=2)

    print("\n=== Persona dataset ===")
    print(f"  base turns:        {stats['persona_base']}")
    print(f"  dominant capped:   {stats['dominant_capped']}")
    print(f"  after weighting:   {stats['persona_weighted']}")
    print(f"  train / val:       {len(p_train)} / {len(p_val)}")
    gif_targets = sum(1 for r in persona if "[gif:" in r["completion"])
    print(f"  gif targets:       {gif_targets} ({gif_targets / len(persona) * 100:.1f}%)")
    avg_ctx = sum(len(r["prompt"]) for r in persona) / len(persona)
    avg_cmp = sum(len(r["completion"]) for r in persona) / len(persona)
    print(f"  avg prompt/compl:  {avg_ctx:.0f} / {avg_cmp:.0f} chars")

    print("\n=== Reaction dataset ===")
    print(f"  emoji vocab:       {stats['emoji_vocab']}")
    print(f"  positive / none:   {stats['react_positive']} / {stats['react_none']}")
    print(f"  train / val:       {len(r_train)} / {len(r_val)}")
    top = list(reaction_freq.items())[:8]
    print(f"  top emoji:         {', '.join(f'{e}={c}' for e, c in top)}")


if __name__ == "__main__":
    main()
