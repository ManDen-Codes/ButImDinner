import glob
import json
import os
import random
import re
from datetime import datetime, timezone

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")

URL_RE = re.compile(r"https?://\S+")


def load_config():
    config_path = os.path.join(ROOT_DIR, "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_all_messages():
    channels = {}
    for path in glob.glob(os.path.join(RAW_DIR, "*.json")):
        with open(path, encoding="utf-8") as f:
            messages = json.load(f)
        if messages:
            channel_name = messages[0]["channel_name"]
            channels[channel_name] = messages
    return channels


def build_user_map(channels):
    user_map = {}
    for messages in channels.values():
        for msg in messages:
            user_map[msg["author_id"]] = msg["author_name"]
    return user_map


def build_channel_map(channels):
    ch_map = {}
    for messages in channels.values():
        if messages:
            ch_map[messages[0]["channel_id"]] = messages[0]["channel_name"]
    return ch_map


def has_url(text):
    return bool(URL_RE.search(text))


def clean_content(content, user_map, channel_map):
    """Replace Discord mention IDs with readable names."""
    def replace_user_mention(match):
        uid = int(match.group(1))
        return f"@{user_map.get(uid, 'unknown')}"

    def replace_channel_mention(match):
        cid = int(match.group(1))
        return f"#{channel_map.get(cid, 'unknown')}"

    content = re.sub(r"<@!?(\d+)>", replace_user_mention, content)
    content = re.sub(r"<#(\d+)>", replace_channel_mention, content)
    content = re.sub(r"<a?:(\w+):\d+>", r":\1:", content)
    return content.strip()


def clean_context_content(content, user_map, channel_map):
    """Clean content for context — replace URLs with [link] instead of dropping."""
    content = clean_content(content, user_map, channel_map)
    content = URL_RE.sub("[link]", content)
    return content.strip()


def merge_consecutive(messages):
    if not messages:
        return []
    merged = [messages[0].copy()]
    for msg in messages[1:]:
        if msg["author_id"] == merged[-1]["author_id"]:
            merged[-1]["content"] += "\n" + msg["content"]
            if msg.get("attachments"):
                merged[-1].setdefault("attachments", []).extend(msg["attachments"])
        else:
            merged.append(msg.copy())
    return merged


def recency_multiplier(timestamp_str, now, weights):
    """Return how many copies of this sample to include based on age."""
    ts = datetime.fromisoformat(timestamp_str)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    months_old = (now - ts).days / 30.44
    for bucket in weights:
        if months_old <= bucket["months"]:
            return bucket["weight"]
    return weights[-1]["weight"]


def build_training_pairs(channels, config, now):
    dinner_id = config["dinner_user_id"]
    context_window = config.get("scraper", {}).get("context_window", 10)
    weights = config.get("recency_weights", [
        {"months": 6,   "weight": 4},
        {"months": 12,  "weight": 3},
        {"months": 24,  "weight": 2},
        {"months": 9999,"weight": 1},
    ])
    user_map = build_user_map(channels)
    channel_map = build_channel_map(channels)

    pairs = []
    skipped_url = 0
    skipped_attachment = 0
    skipped_short = 0

    for channel_name, messages in channels.items():
        msg_by_id = {m["id"]: m for m in messages}

        for i, msg in enumerate(messages):
            if msg["author_id"] != dinner_id:
                continue

            # Drop responses with attachments
            if msg.get("attachments"):
                skipped_attachment += 1
                continue

            cleaned = clean_content(msg["content"], user_map, channel_map)

            # Drop responses with URLs
            if has_url(cleaned):
                skipped_url += 1
                continue

            # Drop responses that are too short after cleaning
            if len(cleaned) < 3:
                skipped_short += 1
                continue

            # Gather preceding context
            start = max(0, i - context_window)
            context_msgs = messages[start:i]

            replied_to = None
            if msg["reply_to_id"] and msg["reply_to_id"] in msg_by_id:
                replied_to = msg_by_id[msg["reply_to_id"]]
                if replied_to not in context_msgs:
                    context_msgs.insert(0, replied_to)

            context_msgs = merge_consecutive(context_msgs)

            context_lines = []
            for ctx in context_msgs:
                ctx_content = clean_context_content(ctx["content"], user_map, channel_map)
                has_attachment = bool(ctx.get("attachments"))

                if ctx_content and has_attachment:
                    ctx_content += " [+attachment]"
                elif has_attachment and not ctx_content:
                    ctx_content = "[shared media]"

                if ctx_content:
                    prefix = "(replied to) " if replied_to and ctx["id"] == replied_to["id"] else ""
                    context_lines.append(f"{prefix}{ctx['author_name']}: {ctx_content}")

            if not context_lines:
                continue

            entry = {
                "messages": [
                    {"role": "system", "content": "You are Dinner. Reply in character."},
                    {"role": "user", "content": "\n".join(context_lines)},
                    {"role": "assistant", "content": cleaned},
                ]
            }

            multiplier = recency_multiplier(msg["timestamp"], now, weights)
            for _ in range(multiplier):
                pairs.append(entry)

    print(f"  Skipped (URL in response):        {skipped_url}")
    print(f"  Skipped (attachment in response): {skipped_attachment}")
    print(f"  Skipped (too short):              {skipped_short}")

    return pairs


def main():
    config = load_config()
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    now = datetime.now(timezone.utc)

    print("Loading raw messages...")
    channels = load_all_messages()
    total_msgs = sum(len(m) for m in channels.values())
    print(f"Loaded {total_msgs} messages from {len(channels)} channels")

    print("Building training pairs...")
    pairs = build_training_pairs(channels, config, now)
    print(f"Built {len(pairs)} training pairs (after weighting)")

    if not pairs:
        print("No training pairs found. Check that dinner_user_id is correct.")
        return

    random.seed(42)
    random.shuffle(pairs)
    split = int(len(pairs) * 0.9)
    train = pairs[:split]
    val = pairs[split:]

    train_path = os.path.join(PROCESSED_DIR, "train.jsonl")
    val_path = os.path.join(PROCESSED_DIR, "val.jsonl")

    for path, data in [(train_path, train), (val_path, val)]:
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nDataset stats:")
    print(f"  Train: {len(train)} pairs -> {train_path}")
    print(f"  Val:   {len(val)} pairs -> {val_path}")

    ctx_lens = [len(p["messages"][1]["content"]) for p in pairs]
    reply_lens = [len(p["messages"][2]["content"]) for p in pairs]
    print(f"  Avg context length: {sum(ctx_lens) / len(ctx_lens):.0f} chars")
    print(f"  Avg reply length:   {sum(reply_lens) / len(reply_lens):.0f} chars")


if __name__ == "__main__":
    main()
