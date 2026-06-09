import asyncio
import json
import os

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


def load_raw_files():
    files = {}
    for filename in os.listdir(RAW_DIR):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(RAW_DIR, filename)
        with open(path, encoding="utf-8") as f:
            messages = json.load(f)
        files[filename] = messages
    return files


def messages_needing_reactions(messages):
    return [
        m for m in messages
        if m.get("reactions_summary")
        and not m.get("reactions")
    ]


async def main():
    config = load_config()
    bot_token = config["bot_token"]

    print("Loading raw data...")
    raw_files = load_raw_files()

    needs_work = {}
    total_messages = 0
    for filename, messages in raw_files.items():
        pending = messages_needing_reactions(messages)
        if pending:
            needs_work[filename] = messages
            total_messages += len(pending)

    if not total_messages:
        print("No messages need reaction user IDs. Nothing to do.")
        return

    print(f"Found {total_messages} messages with reactions across {len(needs_work)} files")

    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        print(f"Logged in as {client.user}")

        for filename, messages in needs_work.items():
            channel_id = messages[0]["channel_id"]
            channel = client.get_channel(channel_id)
            if not channel:
                print(f"\nSkipping {filename} — channel {channel_id} not accessible")
                continue

            pending = messages_needing_reactions(messages)
            msg_id_to_idx = {m["id"]: i for i, m in enumerate(messages)}

            print(f"\nProcessing #{channel.name} — {len(pending)} messages with reactions")

            with tqdm(total=len(pending), desc=f"#{channel.name}", unit=" msgs") as pbar:
                for msg_data in pending:
                    try:
                        discord_msg = await channel.fetch_message(msg_data["id"])
                    except (discord.NotFound, discord.Forbidden, discord.HTTPException) as e:
                        pbar.update(1)
                        continue

                    reactions = []
                    for reaction in discord_msg.reactions:
                        user_ids = []
                        try:
                            async for user in reaction.users():
                                user_ids.append(user.id)
                        except discord.HTTPException:
                            pass
                        reactions.append({
                            "emoji": str(reaction.emoji),
                            "count": reaction.count,
                            "user_ids": user_ids,
                        })

                    idx = msg_id_to_idx[msg_data["id"]]
                    messages[idx]["reactions"] = reactions
                    pbar.update(1)

                    await asyncio.sleep(0.05)

            out_path = os.path.join(RAW_DIR, filename)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(messages, f, ensure_ascii=False, indent=2)
            print(f"  Saved {filename}")

        print("\nDone fetching reaction user IDs!")
        await client.close()

    await client.start(bot_token)


if __name__ == "__main__":
    asyncio.run(main())
