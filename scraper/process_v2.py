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
TENOR_RE = re.compile(r"https://tenor\.com/view/(.+)-(\d+)")
MENTION_RE = re.compile(r"<@!?(\d+)>")
CUSTOM_EMOJI_RE = re.compile(r"<a?:(\w+):\d+>")


def normalize_emoji(emoji):
    """Custom emoji come from Discord as '<:name:id>'; the bot/grammar use ':name:'.
    Normalize so reaction outputs match inference. Unicode emoji pass through."""
    m = CUSTOM_EMOJI_RE.fullmatch(emoji)
    return f":{m.group(1)}:" if m else emoji

SYSTEM_PROMPT = (
    "You are Dinner in a Discord server. Given the conversation context, "
    "decide what to do. Output a JSON action: reply, react, gif, reply_react, "
    "or none. Reply in character."
)


def load_config():
    config_path = os.path.join(ROOT_DIR, "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_dinner_ids(config):
    """All accounts belonging to Dinner (primary + any alt accounts, same person)."""
    ids = {config["dinner_user_id"]}
    ids.update(config.get("dinner_alt_ids") or [])
    return ids


def load_all_messages():
    channels = {}
    for path in glob.glob(os.path.join(RAW_DIR, "*.json")):
        with open(path, encoding="utf-8") as f:
            messages = json.load(f)
        if messages:
            channel_name = messages[0]["channel_name"]
            channels[channel_name] = messages
    return channels


def load_user_index():
    """id -> stable username (from fetch_usernames.py). Empty if not built yet."""
    if not os.path.exists(USER_INDEX_PATH):
        print("  WARNING: user_index.json not found — falling back to scraped display "
              "names. Run scraper/fetch_usernames.py for stable usernames.")
        return {}
    with open(USER_INDEX_PATH, encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f).items()}


def build_user_map(channels, username_index=None):
    # Start from scraped display names, then overlay stable usernames where available.
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
    def replace_user_mention(match):
        uid = int(match.group(1))
        return f"@{user_map.get(uid, 'unknown')}"

    def replace_channel_mention(match):
        cid = int(match.group(1))
        return f"#{channel_map.get(cid, 'unknown')}"

    # Role mentions (<@&id>) have no readable name here and must not teach the model
    # to ping roles; strip them. Must run before the user-mention sub.
    content = re.sub(r"<@&\d+>", "", content)
    content = re.sub(r"<@!?(\d+)>", replace_user_mention, content)
    content = re.sub(r"<#(\d+)>", replace_channel_mention, content)
    content = re.sub(r"<a?:(\w+):\d+>", r":\1:", content)
    return re.sub(r"\s{2,}", " ", content).strip()


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


def recency_multiplier(timestamp_str, now, weights):
    ts = datetime.fromisoformat(timestamp_str)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    months_old = (now - ts).days / 30.44
    for bucket in weights:
        if months_old <= bucket["months"]:
            return bucket["weight"]
    return weights[-1]["weight"]


def build_context_lines(context_msgs, replied_to, user_map, channel_map, dinner_ids):
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
            # Dinner's own messages are labeled "you" so the model recognizes its own
            # turns — matches bot.py's inference-time labeling of the bot's messages.
            # Everyone else uses their stable username (via user_map), not the scraped nick.
            if ctx["author_id"] in dinner_ids:
                name = "you"
            else:
                name = user_map.get(ctx["author_id"], ctx["author_name"])
            lines.append(f"{prefix}{name}: {ctx_content}")
    return "\n".join(lines)


def make_entry(context_text, action_json):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context_text},
            {"role": "assistant", "content": json.dumps(action_json, ensure_ascii=False)},
        ]
    }


def get_dinner_reply_targets(messages, dinner_ids):
    targets = set()
    for i, msg in enumerate(messages):
        if msg["author_id"] not in dinner_ids:
            continue
        if msg["reply_to_id"]:
            targets.add(msg["reply_to_id"])
        for j in range(i - 1, max(0, i - 4), -1):
            if messages[j]["author_id"] not in dinner_ids:
                targets.add(messages[j]["id"])
                break
    return targets


def get_dinner_reacted_ids(messages, dinner_ids):
    reacted = set()
    for msg in messages:
        for r in msg.get("reactions", []):
            if any(d in r.get("user_ids", []) for d in dinner_ids):
                reacted.add(msg["id"])
    return reacted


def get_dinner_reaction_map(messages, dinner_ids):
    reaction_map = {}
    for msg in messages:
        for r in msg.get("reactions", []):
            if any(d in r.get("user_ids", []) for d in dinner_ids):
                if msg["id"] not in reaction_map:
                    reaction_map[msg["id"]] = normalize_emoji(r["emoji"])
    return reaction_map


def build_gif_index(channels, dinner_ids):
    index = {}
    for messages in channels.values():
        for msg in messages:
            if msg["author_id"] not in dinner_ids:
                continue
            match = TENOR_RE.search(msg["content"])
            if not match:
                continue
            slug = match.group(1)
            url = match.group(0)
            words = slug.lower().split("-")
            index[url] = words
    return index


def build_all_pairs(channels, config, now, username_index=None):
    dinner_primary = config["dinner_user_id"]
    dinner_ids = get_dinner_ids(config)
    context_window = config.get("scraper", {}).get("context_window", 10)
    weights = config.get("recency_weights", [
        {"months": 6, "weight": 4},
        {"months": 12, "weight": 3},
        {"months": 24, "weight": 2},
        {"months": 9999, "weight": 1},
    ])
    gif_oversample = config.get("training", {}).get("gif_oversample", 4)
    none_ratio = config.get("training", {}).get("none_ratio", 0.35)

    user_map = build_user_map(channels, username_index)
    # Normalize all of Dinner's accounts to one identity so mentions of either his old
    # or current account render as the same name (and the model sees one persona).
    primary_name = user_map.get(dinner_primary, "dinnerlore")
    for did in dinner_ids:
        user_map[did] = primary_name
    channel_map = build_channel_map(channels)

    reply_pairs = []
    react_pairs = []
    reply_react_pairs = []
    gif_pairs = []
    none_candidates = []

    stats = defaultdict(int)

    for channel_name, messages in channels.items():
        msg_by_id = {m["id"]: m for m in messages}
        dinner_reply_targets = get_dinner_reply_targets(messages, dinner_ids)
        dinner_reacted_ids = get_dinner_reacted_ids(messages, dinner_ids)
        dinner_reaction_map = get_dinner_reaction_map(messages, dinner_ids)

        # --- reply, gif, reply_react pairs ---
        for i, msg in enumerate(messages):
            if msg["author_id"] not in dinner_ids:
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

            context_text = build_context_lines(context_msgs, replied_to, user_map, channel_map, dinner_ids)
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

            context_text = build_context_lines(context_msgs, None, user_map, channel_map, dinner_ids)
            if not context_text:
                continue

            action = {"action": "react", "emoji": emoji}
            multiplier = recency_multiplier(msg["timestamp"], now, weights)
            for _ in range(multiplier):
                react_pairs.append(make_entry(context_text, action))
            stats["react"] += 1

        # --- none candidates ---
        dinner_active = any(m["author_id"] in dinner_ids for m in messages)
        if not dinner_active:
            continue

        for i, msg in enumerate(messages):
            if msg["author_id"] in dinner_ids:
                continue
            if msg["id"] in dinner_reply_targets:
                continue
            if msg["id"] in dinner_reacted_ids:
                continue
            if not msg["content"].strip():
                continue
            none_candidates.append((channel_name, i, msg))

    # Cap react-only pairs relative to reply pairs so the model doesn't default to reacting.
    react_cap_ratio = config.get("training", {}).get("react_cap_ratio", 0.4)
    max_react = int(len(reply_pairs) * react_cap_ratio)
    if len(react_pairs) > max_react:
        random.seed(42)
        react_pairs = random.sample(react_pairs, max_react)
        stats["react_capped"] = max_react

    # Sample none pairs.
    # NOTE: append each sampled candidate ONCE (no recency multiplier). action_count is
    # already recency-weighted, so multiplying none again would inflate it far past
    # none_ratio (the original bug: 0.35 configured -> ~55% actual none).
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

        context_text = build_context_lines(context_msgs, None, user_map, channel_map, dinner_ids)
        if not context_text:
            continue

        none_pairs.append(make_entry(context_text, {"action": "none"}))

    stats["none"] = len(none_pairs)

    print(f"\n  Action counts (before recency weighting):")
    print(f"    reply:       {stats['reply']}")
    print(f"    react:       {stats['react']}")
    if stats.get("react_capped"):
        print(f"      (react-only capped to {stats['react_capped']} weighted pairs "
              f"= {react_cap_ratio:.0%} of reply)")
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

    cutoff_date_str = config.get("scraper", {}).get("cutoff_date")
    if cutoff_date_str:
        cutoff_dt = datetime.fromisoformat(cutoff_date_str).replace(tzinfo=timezone.utc)
        for ch in list(channels.keys()):
            before = len(channels[ch])
            def _ts(m):
                ts = datetime.fromisoformat(m["timestamp"])
                return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            channels[ch] = [m for m in channels[ch] if _ts(m) <= cutoff_dt]
            after = len(channels[ch])
            if before != after:
                print(f"  #{ch}: filtered {before - after} messages after {cutoff_date_str}")
        print(f"Cutoff date applied: {cutoff_date_str}")

    total_msgs = sum(len(m) for m in channels.values())
    print(f"Loaded {total_msgs} messages from {len(channels)} channels")

    print("Building GIF index...")
    gif_index = build_gif_index(channels, get_dinner_ids(config))
    gif_index_path = os.path.join(PROCESSED_DIR, "gif_index.json")
    with open(gif_index_path, "w", encoding="utf-8") as f:
        json.dump(gif_index, f, ensure_ascii=False, indent=2)
    print(f"  {len(gif_index)} unique GIFs -> {gif_index_path}")

    print("Loading username index...")
    username_index = load_user_index()
    print(f"  {len(username_index)} usernames loaded")

    print("Building v2 training pairs...")
    random.seed(42)
    pairs = build_all_pairs(channels, config, now, username_index)
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
