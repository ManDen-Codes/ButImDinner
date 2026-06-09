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

URL_RE = re.compile(r"https?://\S+")
TENOR_RE = re.compile(r"https://tenor\.com/view/(.+)-(\d+)")
MENTION_RE = re.compile(r"<@!?(\d+)>")

SYSTEM_PROMPT = (
    "You are Dinner in a Discord server. Given the conversation context, "
    "decide what to do. Output a JSON action: reply, react, gif, reply_react, "
    "or none. Reply in character."
)


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


def clean_content(content, user_map, channel_map):
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
    content = clean_content(content, user_map, channel_map)
    content = URL_RE.sub("[link]", content)
    return content.strip()


def has_url(text):
    return bool(URL_RE.search(text))


def extract_mentions(raw_content, user_map):
    mention_ids = MENTION_RE.findall(raw_content)
    mentions = []
    for uid_str in mention_ids:
        uid = int(uid_str)
        if uid in user_map:
            name = user_map[uid]
            if name not in mentions:
                mentions.append(name)
    return mentions


def extract_tenor_query(content):
    match = TENOR_RE.search(content)
    if not match:
        return None
    slug = match.group(1)
    words = slug.split("-")
    if words:
        return " ".join(words)
    return None


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
    ts = datetime.fromisoformat(timestamp_str)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    months_old = (now - ts).days / 30.44
    for bucket in weights:
        if months_old <= bucket["months"]:
            return bucket["weight"]
    return weights[-1]["weight"]


def build_context_lines(context_msgs, replied_to, user_map, channel_map):
    context_msgs = merge_consecutive(context_msgs)
    lines = []
    for ctx in context_msgs:
        ctx_content = clean_context_content(ctx["content"], user_map, channel_map)
        has_attachment = bool(ctx.get("attachments"))

        if ctx_content and has_attachment:
            ctx_content += " [+attachment]"
        elif has_attachment and not ctx_content:
            ctx_content = "[shared media]"

        if ctx_content:
            prefix = "(replied to) " if replied_to and ctx["id"] == replied_to["id"] else ""
            lines.append(f"{prefix}{ctx['author_name']}: {ctx_content}")
    return "\n".join(lines)


def make_entry(context_text, action_json):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context_text},
            {"role": "assistant", "content": json.dumps(action_json, ensure_ascii=False)},
        ]
    }


def get_dinner_reply_targets(messages, dinner_id):
    targets = set()
    for i, msg in enumerate(messages):
        if msg["author_id"] != dinner_id:
            continue
        if msg["reply_to_id"]:
            targets.add(msg["reply_to_id"])
        for j in range(i - 1, max(0, i - 4), -1):
            if messages[j]["author_id"] != dinner_id:
                targets.add(messages[j]["id"])
                break
    return targets


def get_dinner_reacted_ids(messages, dinner_id):
    reacted = set()
    for msg in messages:
        for r in msg.get("reactions", []):
            if dinner_id in r.get("user_ids", []):
                reacted.add(msg["id"])
    return reacted


def get_dinner_reaction_map(messages, dinner_id):
    reaction_map = {}
    for msg in messages:
        for r in msg.get("reactions", []):
            if dinner_id in r.get("user_ids", []):
                if msg["id"] not in reaction_map:
                    reaction_map[msg["id"]] = r["emoji"]
    return reaction_map


def build_all_pairs(channels, config, now):
    dinner_id = config["dinner_user_id"]
    context_window = config.get("scraper", {}).get("context_window", 10)
    weights = config.get("recency_weights", [
        {"months": 6, "weight": 4},
        {"months": 12, "weight": 3},
        {"months": 24, "weight": 2},
        {"months": 9999, "weight": 1},
    ])
    gif_oversample = config.get("training", {}).get("gif_oversample", 4)
    none_ratio = config.get("training", {}).get("none_ratio", 0.35)

    user_map = build_user_map(channels)
    channel_map = build_channel_map(channels)

    reply_pairs = []
    react_pairs = []
    reply_react_pairs = []
    gif_pairs = []
    none_candidates = []

    stats = defaultdict(int)

    for channel_name, messages in channels.items():
        msg_by_id = {m["id"]: m for m in messages}
        dinner_reply_targets = get_dinner_reply_targets(messages, dinner_id)
        dinner_reacted_ids = get_dinner_reacted_ids(messages, dinner_id)
        dinner_reaction_map = get_dinner_reaction_map(messages, dinner_id)

        # --- reply, gif, reply_react pairs ---
        for i, msg in enumerate(messages):
            if msg["author_id"] != dinner_id:
                continue

            cleaned = clean_content(msg["content"], user_map, channel_map)
            mentions = extract_mentions(msg["content"], user_map)
            tenor_query = extract_tenor_query(msg["content"])
            has_attachment = bool(msg.get("attachments"))

            start = max(0, i - context_window)
            context_msgs = messages[start:i]

            replied_to = None
            if msg["reply_to_id"] and msg["reply_to_id"] in msg_by_id:
                replied_to = msg_by_id[msg["reply_to_id"]]
                if replied_to not in context_msgs:
                    context_msgs.insert(0, replied_to)

            context_text = build_context_lines(context_msgs, replied_to, user_map, channel_map)
            if not context_text:
                continue

            multiplier = recency_multiplier(msg["timestamp"], now, weights)

            # GIF action
            if tenor_query:
                action = {"action": "gif", "query": tenor_query}
                for _ in range(multiplier * gif_oversample):
                    gif_pairs.append(make_entry(context_text, action))
                stats["gif"] += 1
                continue

            # Skip non-text responses
            if has_attachment:
                stats["skipped_attachment"] += 1
                continue
            if has_url(cleaned):
                stats["skipped_url"] += 1
                continue
            if len(cleaned) < 3:
                stats["skipped_short"] += 1
                continue

            # Check if Dinner also reacted to a message in this context window
            reply_target_id = msg["reply_to_id"]
            reacted_emoji = None
            check_ids = [reply_target_id] if reply_target_id else []
            for j in range(max(0, i - 3), i):
                check_ids.append(messages[j]["id"])
            for check_id in check_ids:
                if check_id and check_id in dinner_reaction_map:
                    reacted_emoji = dinner_reaction_map[check_id]
                    break

            action = {"action": "reply", "text": cleaned, "mentions": mentions}

            if reacted_emoji:
                action = {
                    "action": "reply_react",
                    "text": cleaned,
                    "emoji": reacted_emoji,
                    "mentions": mentions,
                }
                for _ in range(multiplier):
                    reply_react_pairs.append(make_entry(context_text, action))
                stats["reply_react"] += 1
            else:
                for _ in range(multiplier):
                    reply_pairs.append(make_entry(context_text, action))
                stats["reply"] += 1

        # --- react-only pairs (Dinner reacted but didn't reply) ---
        for i, msg in enumerate(messages):
            if msg["id"] not in dinner_reaction_map:
                continue
            if msg["id"] in dinner_reply_targets:
                continue

            emoji = dinner_reaction_map[msg["id"]]
            start = max(0, i - context_window)
            context_msgs = messages[start:i + 1]

            context_text = build_context_lines(context_msgs, None, user_map, channel_map)
            if not context_text:
                continue

            action = {"action": "react", "emoji": emoji}
            multiplier = recency_multiplier(msg["timestamp"], now, weights)
            for _ in range(multiplier):
                react_pairs.append(make_entry(context_text, action))
            stats["react"] += 1

        # --- none candidates ---
        dinner_active = any(m["author_id"] == dinner_id for m in messages)
        if not dinner_active:
            continue

        for i, msg in enumerate(messages):
            if msg["author_id"] == dinner_id:
                continue
            if msg["id"] in dinner_reply_targets:
                continue
            if msg["id"] in dinner_reacted_ids:
                continue
            if not msg["content"].strip():
                continue
            none_candidates.append((channel_name, i, msg))

    # Sample none pairs
    action_count = len(reply_pairs) + len(react_pairs) + len(reply_react_pairs) + len(gif_pairs)
    target_none = int(action_count * none_ratio / (1 - none_ratio))
    target_none = min(target_none, len(none_candidates))

    random.seed(42)
    sampled_nones = random.sample(none_candidates, target_none)

    none_pairs = []
    for channel_name, i, msg in sampled_nones:
        messages = channels[channel_name]
        start = max(0, i - context_window)
        context_msgs = messages[start:i + 1]

        user_map_local = build_user_map(channels)
        channel_map_local = build_channel_map(channels)
        context_text = build_context_lines(context_msgs, None, user_map_local, channel_map_local)
        if not context_text:
            continue

        action = {"action": "none"}
        multiplier = recency_multiplier(msg["timestamp"], now, weights)
        for _ in range(multiplier):
            none_pairs.append(make_entry(context_text, action))

    stats["none"] = len(sampled_nones)

    print(f"\n  Action counts (before recency weighting):")
    print(f"    reply:       {stats['reply']}")
    print(f"    react:       {stats['react']}")
    print(f"    reply_react: {stats['reply_react']}")
    print(f"    gif:         {stats['gif']} (x{gif_oversample} oversample)")
    print(f"    none:        {stats['none']} (sampled from {len(none_candidates)} candidates)")
    print(f"  Skipped:")
    print(f"    URL in response:        {stats['skipped_url']}")
    print(f"    Attachment in response:  {stats['skipped_attachment']}")
    print(f"    Too short:              {stats['skipped_short']}")

    all_pairs = reply_pairs + react_pairs + reply_react_pairs + gif_pairs + none_pairs
    return all_pairs


def stratified_split(pairs, train_ratio=0.9):
    by_action = defaultdict(list)
    for pair in pairs:
        action_data = json.loads(pair["messages"][2]["content"])
        by_action[action_data["action"]].append(pair)

    train, val = [], []
    for action_type, action_pairs in by_action.items():
        random.shuffle(action_pairs)
        split = int(len(action_pairs) * train_ratio)
        train.extend(action_pairs[:split])
        val.extend(action_pairs[split:])

    random.shuffle(train)
    random.shuffle(val)
    return train, val


def main():
    config = load_config()
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    now = datetime.now(timezone.utc)

    print("Loading raw messages...")
    channels = load_all_messages()
    total_msgs = sum(len(m) for m in channels.values())
    print(f"Loaded {total_msgs} messages from {len(channels)} channels")

    print("Building v2 training pairs...")
    random.seed(42)
    pairs = build_all_pairs(channels, config, now)
    print(f"\nTotal pairs (after weighting): {len(pairs)}")

    if not pairs:
        print("No training pairs found. Check that dinner_user_id is correct.")
        return

    random.seed(42)
    train, val = stratified_split(pairs)

    train_path = os.path.join(PROCESSED_DIR, "train.jsonl")
    val_path = os.path.join(PROCESSED_DIR, "val.jsonl")

    for path, data in [(train_path, train), (val_path, val)]:
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Stats
    action_counts = defaultdict(int)
    for pair in pairs:
        action_data = json.loads(pair["messages"][2]["content"])
        action_counts[action_data["action"]] += 1

    print(f"\nDataset stats:")
    print(f"  Train: {len(train)} pairs -> {train_path}")
    print(f"  Val:   {len(val)} pairs -> {val_path}")
    print(f"\n  Action distribution (after weighting):")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        pct = count / len(pairs) * 100
        print(f"    {action:12s}: {count:6d} ({pct:.1f}%)")

    ctx_lens = [len(p["messages"][1]["content"]) for p in pairs]
    reply_lens = [len(p["messages"][2]["content"]) for p in pairs]
    print(f"\n  Avg context length: {sum(ctx_lens) / len(ctx_lens):.0f} chars")
    print(f"  Avg action length:  {sum(reply_lens) / len(reply_lens):.0f} chars")


if __name__ == "__main__":
    main()
