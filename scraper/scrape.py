import asyncio
import json
import os
import sys

import discord
import yaml
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")


def load_config():
    config_path = os.path.join(ROOT_DIR, "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


async def scrape_channel(channel, pbar):
    messages = []
    async for message in channel.history(limit=None, oldest_first=True):
        msg_data = {
            "id": message.id,
            "author_id": message.author.id,
            "author_name": message.author.display_name,
            "content": message.content,
            "timestamp": message.created_at.isoformat(),
            "channel_id": channel.id,
            "channel_name": channel.name,
            "attachments": [a.url for a in message.attachments],
            "reply_to_id": message.reference.message_id if message.reference else None,
        }
        messages.append(msg_data)
        pbar.update(1)
    return messages


async def main():
    config = load_config()
    bot_token = config["bot_token"]
    guild_id = config["guild_id"]

    os.makedirs(RAW_DIR, exist_ok=True)

    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True

    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        print(f"Logged in as {client.user}")
        guild = client.get_guild(guild_id)
        if not guild:
            print(f"Could not find guild {guild_id}")
            await client.close()
            return

        text_channels = [
            ch for ch in guild.channels if isinstance(ch, discord.TextChannel)
        ]
        print(f"Found {len(text_channels)} text channels in '{guild.name}'")

        for channel in text_channels:
            out_path = os.path.join(RAW_DIR, f"{channel.id}_{channel.name}.json")
            if os.path.exists(out_path):
                print(f"\nSkipping #{channel.name} (already scraped)")
                continue

            print(f"\nScraping #{channel.name}...")
            try:
                with tqdm(desc=f"#{channel.name}", unit=" msgs") as pbar:
                    messages = await scrape_channel(channel, pbar)

                if messages:
                    out_path = os.path.join(RAW_DIR, f"{channel.id}_{channel.name}.json")
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(messages, f, ensure_ascii=False, indent=2)
                    print(f"  Saved {len(messages)} messages")
                else:
                    print(f"  No messages (or no access)")
            except discord.Forbidden:
                print(f"  Skipped (no permission)")
            except Exception as e:
                print(f"  Error: {e}")

        print("\nDone scraping!")
        await client.close()

    await client.start(bot_token)


if __name__ == "__main__":
    asyncio.run(main())
