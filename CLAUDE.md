# ButImDinner

Discord bot that impersonates a friend ("Dinner") by fine-tuning a local LLM on their Discord message history. The bot uses a structured action system — the model outputs JSON deciding whether to reply, react, send a GIF, combo, or stay silent.

## Architecture

```
scraper → data processing → QLoRA fine-tuning → GGUF export → discord bot (local inference)
```

**Stack**: Python, discord.py, pure HuggingFace (transformers + peft + trl + bitsandbytes), llama-cpp-python (inference)
**Base model**: Qwen3.5 9B (`Qwen/Qwen3.5-9B`)
**Training machine**: RTX 4070 Ti (12GB VRAM), QLoRA 4-bit, bf16
**Inference machine**: ASUS laptop — Intel Core Ultra 285H, Intel Arc 140T iGPU, 32GB RAM — runs GGUF via llama-cpp-python (Vulkan backend); or any machine with the merged HF model

## Project Structure

- `config.example.yaml` — template config (copy to `config.yaml` and fill in)
- `scraper/scrape.py` — connects as Discord bot, exports all messages from a server to `data/raw/` (includes reaction metadata)
- `scraper/scrape_reactions.py` — second-pass scraper that fetches reaction user IDs for messages with reactions
- `scraper/process.py` — v1 processor: converts raw messages into plain text ChatML training pairs
- `scraper/process_v2.py` — v2 processor: generates action-labeled JSON training pairs (reply, react, gif, reply_react, none)
- `training/train.py` — QLoRA fine-tune using pure HuggingFace stack
- `training/eval.py` — standalone eval script (runs on a checkpoint without OOMing; use instead of in-training eval)
- `training/export.py` — merges LoRA adapter into base model, saves to `data/model/merged/`
- `bot/bot.py` — Discord bot with unified action system; auto-detects backend (HF if model_path is a directory, GGUF if .gguf file)

## Workflow

1. `python scraper/scrape.py` — scrape messages (needs `config.yaml` with bot token, guild ID, Dinner's user ID)
2. `python scraper/scrape_reactions.py` — fetch reaction user IDs (second pass, only messages with reactions)
3. `python scraper/process_v2.py` — build action-labeled training pairs from raw data
4. `python training/train.py` — fine-tune (run on GPU machine); resumes automatically from latest checkpoint
5. `python training/export.py` — merge LoRA into base model → `data/model/merged/`; optionally convert to GGUF with llama.cpp
6. `python bot/bot.py` — run the bot

## Current Status

- v1 training complete (Qwen3 4B, 5 epochs, ~29k pairs — plain text replies only)
- v2 training complete (Qwen3.5 9B, 3 epochs, ~54k pairs — full action system)
- v2 deployed: `dinner_q8.gguf` running on ASUS laptop via Vulkan

## Key Details

- `data/` is gitignored — transfer it between machines manually
- **Unsloth was abandoned** — TRL 0.24.0 incompatibility with `<EOS_TOKEN>` placeholder; pure HF stack used instead
- **Import order**: `import torch` must be LAST import in train.py or `datasets` crashes silently on Windows
- **Thinking mode**: Qwen3.5 has built-in chain-of-thought (`<think>` tags); disabled via `enable_thinking=False` (HF) and `/no_think` in system prompt (GGUF); `strip_thinking()` removes any leaked `<think>` blocks from output
- **Bot backend auto-detection**: `model_path` in config — if directory → HF transformers; if `.gguf` file → llama-cpp-python
- **v2 action system**: model outputs JSON with action type (reply, react, gif, reply_react, none) — single model call decides what to do; bot dispatches accordingly
- **v2 data labeling**: `process_v2.py` generates action-labeled training pairs; GIF queries extracted from Tenor URL slugs; "none" actions sampled from messages Dinner ignored; `mentions` field extracted from `<@id>` patterns
- **Data processing filters**: URL-containing responses dropped, attachment-only responses dropped, responses < 3 chars dropped; URLs in context replaced with `[link]`; recency weighting duplicates recent messages (configurable in `recency_weights` config); GIF oversampling configurable via `gif_oversample`
- **Scrape cutoff date**: `scraper.cutoff_date` in config excludes messages after a given date — use to prevent bot-generated messages from polluting training data after a v1 deployment
- **Mention handling**: `@mentions` in input are resolved to readable names (bot's own mention → configurable `mention_name`, others → username); bot's own messages in context labeled as `you:` so the model knows which are its own; output `mentions` field resolved to real Discord mentions
- **Engagement triggers**: always engages when someone replies to its message (with `reply_chain_decay` tapering); optionally always engages on @mention (`reply_on_mention` config); otherwise rolls `engagement_chance` in allowed channels — model decides the action type
- **GBNF grammar**: constrains GGUF output to valid JSON matching the action schema; grammar built dynamically per-guild to include server custom emoji names; cached and rebuilt on emoji changes
- **GIF support**: `gif` action matches model's query against Dinner's own GIF history (local index at `data/processed/gif_index.json`, built by `process_v2.py`); no external API needed
- **Output mention resolution**: if the model outputs `@name`, it's matched against guild members (username and nickname) and converted to a real Discord mention; `@everyone`/`@here` pings are suppressed
- **Discord intents**: requires `message_content` and `members` intents enabled in Discord Developer Portal
- Config is in `config.yaml` (gitignored) — copy from `config.example.yaml`
- Zero-cost solution: all open-source, runs fully local, no API calls

## Training Dependencies (GPU machine)

```
pip install torch --index-url https://download.pytorch.org/whl/cu130 --force-reinstall
pip install datasets transformers peft trl==0.24.0 bitsandbytes pyyaml
pip uninstall torchvision -y
```

Note: `trl` must be pinned to `0.24.0` — newer versions have a Windows Unicode bug reading Jinja templates.

**Training quirks for Qwen3.5-9B on 12GB VRAM:**
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (already in train.py) to prevent fragmentation OOM
- Do NOT use `gradient_checkpointing=True` — it's 10x slower on Qwen3.5's hybrid architecture
- Set `eval_strategy="no"` in train.py — eval OOMs at 12GB; use `training/eval.py` separately instead
- `save_total_limit=3` keeps only the 3 most recent checkpoints to manage disk space

## Bot Dependencies

**On training machine (HF backend, no GGUF needed):**
```
pip install discord.py pyyaml transformers torch accelerate
```

**On laptop (GGUF + Vulkan backend):**
```
pip install discord.py pyyaml
pip install <vulkan wheel from https://github.com/abetlen/llama-cpp-python/releases>
```
Download the `llama_cpp_python-*-py3-none-win_amd64.whl` from the `vulkan` release tag.

## Laptop Setup (ASUS Intel Arc 140T)

1. Transfer `data/model/dinner_q8.gguf` and `data/processed/gif_index.json` from GPU machine
2. Transfer `config.yaml` (gitignored)
3. Clone repo
4. Install dependencies (above)
5. Set `model_path: "data/model/dinner_q8.gguf"` in config.yaml
6. `python bot/bot.py`
