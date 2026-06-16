"""Build an id -> username map for every user who appears in the scraped data.

The scraper only stored display names (nicknames), which drift over time. Usernames
are stable global handles, so we use them for speaker labels and mention resolution in
both training (process_v2.py) and inference (bot.py). Run this once after scraping.

Output: data/processed/user_index.json  ({ "<author_id>": "<username>" })
"""
import asyncio
import glob
import json
import os

import discord
import yaml
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
OUT_PATH = os.path.join(PROCESSED_DIR, "user_index.json")


def load_config():
    with open(os.path.join(ROOT_DIR, "config.yaml")) as f:
        return yaml.safe_load(f)


def collect_users():
    """Return {author_id: fallback_display_name} for every user in the raw data."""
    users = {}
    for path in glob.glob(os.path.join(RAW_DIR, "*.json")):
        with open(path, encoding="utf-8") as f:
            messages = json.load(f)
        for m in messages:
            users.setdefault(m["author_id"], m["author_name"])
    return users


async def main():
    config = load_config()
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    users = collect_users()
    print(f"Found {len(users)} unique users in raw data")

    intents = discord.Intents.default()
    intents.members = True
    client = discord.Client(intents=intents)

    user_index = {}

    @client.event
    async def on_ready():
        print(f"Logged in as {client.user}")
        for uid, fallback in tqdm(users.items(), desc="Fetching usernames", unit=" users"):
            try:
                user = await client.fetch_user(uid)
                user_index[str(uid)] = user.name
            except (discord.NotFound, discord.HTTPException):
                # Deleted account / webhook — fall back to the scraped display name.
                user_index[str(uid)] = fallback
            await asyncio.sleep(0.1)

        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(user_index, f, ensure_ascii=False, indent=2)
        print(f"\nSaved {len(user_index)} usernames -> {OUT_PATH}")
        await client.close()

    await client.start(config["bot_token"])


if __name__ == "__main__":
    asyncio.run(main())
