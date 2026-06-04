# ButImDinner

Discord bot that impersonates a friend ("Dinner") by fine-tuning a local LLM on their Discord message history. The bot randomly replies to messages in allowlisted channels.

## Architecture

```
scraper → data processing → QLoRA fine-tuning → GGUF export → discord bot (local inference)
```

**Stack**: Python, discord.py, Unsloth (fine-tuning), llama-cpp-python (inference)
**Base model**: Llama 3.1 8B (4-bit quantized via QLoRA)
**Target GPU**: RTX 4070 Ti (12GB VRAM)

## Project Structure

- `config.example.yaml` — template config (copy to `config.yaml` and fill in)
- `scraper/scrape.py` — connects as Discord bot, exports all messages from a server to `data/raw/`
- `scraper/process.py` — converts raw messages into ChatML-style training pairs (JSONL) in `data/processed/`
- `training/train.py` — QLoRA fine-tune Llama 3.1 8B using Unsloth
- `training/export.py` — merges LoRA adapter and exports to GGUF (Q4_K_M)
- `bot/bot.py` — Discord bot that loads the GGUF model and randomly replies in Dinner's style

## Workflow

1. `python scraper/scrape.py` — scrape messages (needs `config.yaml` with bot token, guild ID, Dinner's user ID)
2. `python scraper/process.py` — build training pairs from raw data
3. `python training/train.py` — fine-tune the model (run on GPU machine)
4. `python training/export.py` — export to GGUF
5. `python bot/bot.py` — run the bot

## Current Status

- Scraping is in progress / completed
- Next step: run `process.py`, then train on the GPU machine
- Training dependencies (install on GPU machine): `pip install unsloth torch transformers datasets trl bitsandbytes pyyaml`
- Bot dependencies: `pip install discord.py pyyaml llama-cpp-python`

## Key Details

- `data/` is gitignored — transfer it to the GPU machine separately
- The data processor handles Discord replies by pulling the replied-to message into context and tagging it with `(replied to)`
- Config is in `config.yaml` (gitignored) — copy from `config.example.yaml`
- Zero-cost solution: all open-source, runs fully local, no API calls
- Llama 3.1 requires a free HuggingFace account and accepting Meta's license
