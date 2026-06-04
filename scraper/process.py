import glob
import json
import os
import random
import re

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")


def load_config():
    config_path = os.path.join(ROOT_DIR, "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_all_messages():
    """Load and merge all raw channel JSONs into one chronological list per channel."""
    channels = {}
    for path in glob.glob(os.path.join(RAW_DIR, "*.json")):
        with open(path, encoding="utf-8") as f:
            messages = json.load(f)
        if messages:
            channel_name = messages[0]["channel_name"]
            channels[channel_name] = messages
    return channels


def build_user_map(channels):
    """Build a mapping of user ID -> display name from all messages."""
    user_map = {}
    for messages in channels.values():
        for msg in messages:
            user_map[msg["author_id"]] = msg["author_name"]
    return user_map


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
    # Strip custom emoji to just the name
    content = re.sub(r"<a?:(\w+):\d+>", r":\1:", content)
    return content.strip()


def build_channel_map(channels):
    """Build channel ID -> name mapping."""
    ch_map = {}
    for messages in channels.values():
        if messages:
            ch_map[messages[0]["channel_id"]] = messages[0]["channel_name"]
    return ch_map


def merge_consecutive(messages):
    """Merge consecutive messages from the same author into one."""
    if not messages:
        return []
    merged = [messages[0].copy()]
    for msg in messages[1:]:
        if msg["author_id"] == merged[-1]["author_id"]:
            merged[-1]["content"] += "\n" + msg["content"]
        else:
            merged.append(msg.copy())
    return merged


def build_training_pairs(channels, config):
    """Build (context, response) pairs for every Dinner message."""
    dinner_id = config["dinner_user_id"]
    context_window = config.get("scraper", {}).get("context_window", 10)
    user_map = build_user_map(channels)
    channel_map = build_channel_map(channels)

    pairs = []

    for channel_name, messages in channels.items():
        msg_by_id = {m["id"]: m for m in messages}

        for i, msg in enumerate(messages):
            if msg["author_id"] != dinner_id:
                continue
            cleaned = clean_content(msg["content"], user_map, channel_map)
            if not cleaned:
                continue

            # Gather preceding context
            start = max(0, i - context_window)
            context_msgs = messages[start:i]

            # If Dinner replied to a specific message, ensure it's in context
            replied_to = None
            if msg["reply_to_id"] and msg["reply_to_id"] in msg_by_id:
                replied_to = msg_by_id[msg["reply_to_id"]]
                if replied_to not in context_msgs:
                    context_msgs.insert(0, replied_to)

            context_msgs = merge_consecutive(context_msgs)

            context_lines = []
            for ctx in context_msgs:
                ctx_content = clean_content(ctx["content"], user_map, channel_map)
                if ctx_content:
                    prefix = "(replied to) " if replied_to and ctx["id"] == replied_to["id"] else ""
                    context_lines.append(f"{prefix}{ctx['author_name']}: {ctx_content}")

            if not context_lines:
                continue

            pairs.append({
                "messages": [
                    {
                        "role": "system",
                        "content": "You are Dinner. Reply in character.",
                    },
                    {"role": "user", "content": "\n".join(context_lines)},
                    {"role": "assistant", "content": cleaned},
                ]
            })

    return pairs


def main():
    config = load_config()
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Loading raw messages...")
    channels = load_all_messages()
    total_msgs = sum(len(m) for m in channels.values())
    print(f"Loaded {total_msgs} messages from {len(channels)} channels")

    print("Building training pairs...")
    pairs = build_training_pairs(channels, config)
    print(f"Built {len(pairs)} training pairs")

    if not pairs:
        print("No training pairs found. Check that dinner_user_id is correct.")
        return

    # Shuffle and split 90/10
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

    # Compute average lengths
    ctx_lens = [len(p["messages"][1]["content"]) for p in pairs]
    reply_lens = [len(p["messages"][2]["content"]) for p in pairs]
    print(f"  Avg context length: {sum(ctx_lens) / len(ctx_lens):.0f} chars")
    print(f"  Avg reply length:   {sum(reply_lens) / len(reply_lens):.0f} chars")


if __name__ == "__main__":
    main()
