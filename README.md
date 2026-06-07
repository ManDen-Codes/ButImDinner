# ButImDinner

A Discord bot that impersonates a friend by fine-tuning a local LLM on their message history. The bot lurks in allowlisted channels and randomly replies in their style — fully local, no API costs.

## How it works

```
scraper → data processing → QLoRA fine-tuning → GGUF export → discord bot
```

- **Scraper**: logs in as a Discord bot and exports all messages from a server
- **Processor**: converts raw logs into ChatML training pairs (context → Dinner's reply)
- **Training**: QLoRA fine-tunes Qwen3.5 4B on your GPU using Unsloth
- **Export**: merges the LoRA adapter and quantizes to GGUF (Q4_K_M, ~2.5GB)
- **Bot**: runs the GGUF locally and randomly replies in character

## Requirements

### Hardware
- **Training**: NVIDIA GPU with 8GB+ VRAM (tested on RTX 4070 Ti 12GB)
- **Inference**: any machine with 4GB+ RAM — CPU-only works, GPU optional

### Software
- Python 3.10+
- CUDA 12.x (training only)

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd ButImDinner
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Configure

```bash
cp config.example.yaml config.yaml
```

Edit `config.yaml` and fill in:
- `bot_token` — Discord bot token ([Discord Developer Portal](https://discord.com/developers/applications))
- `guild_id` — the server ID to scrape (right-click server → Copy Server ID with developer mode on)
- `dinner_user_id` — the target user's Discord ID (right-click user → Copy User ID)
- `bot.allowed_channels` — list of channel IDs the bot is allowed to reply in

---

## Step 1 — Scrape

Install dependencies:

```bash
pip install discord.py pyyaml tqdm
```

Run the scraper:

```bash
python scraper/scrape.py
```

This exports all messages from the server to `data/raw/` as JSON files (one per channel). The bot needs **Read Message History** permission in each channel.

---

## Step 2 — Process

No new dependencies needed.

```bash
python scraper/process.py
```

Reads `data/raw/`, builds training pairs for every message the target user sent, and writes:
- `data/processed/train.jsonl` (90%)
- `data/processed/val.jsonl` (10%)

Each example is a ChatML conversation: the preceding chat context as the user turn, and the target's reply as the assistant turn. Reply context (`(replied to)`) is handled automatically.

---

## Step 3 — Train

Install dependencies (GPU machine):

```bash
pip install "unsloth[cu124-torch260]" transformers datasets trl bitsandbytes pyyaml
```

> Adjust the `unsloth` extra to match your CUDA version. See [Unsloth installation docs](https://unsloth.ai/docs/get-started/installing-unsloth) for options.

Run training:

```bash
python training/train.py
```

This downloads Qwen3.5 4B (~2.5GB from HuggingFace, no account required), applies QLoRA, and trains for 3 epochs. The LoRA adapter is saved to `data/model/lora_adapter/`.

Training config can be adjusted in `config.yaml` under the `training:` key:

| Key | Default | Notes |
|-----|---------|-------|
| `base_model` | `unsloth/Qwen3.5-4B` | HuggingFace model ID |
| `epochs` | `3` | increase for more data |
| `batch_size` | `2` | lower if OOM |
| `gradient_accumulation_steps` | `4` | effective batch = batch × accum |
| `learning_rate` | `0.0002` | |
| `lora_rank` | `64` | |
| `max_seq_length` | `2048` | |

---

## Step 4 — Export to GGUF

No new dependencies needed (uses the same training environment).

```bash
python training/export.py
```

Merges the LoRA adapter into the base model and exports a Q4_K_M GGUF to `data/model/dinner.gguf`.

---

## Step 5 — Run the bot

Install dependencies (inference machine):

```bash
pip install discord.py pyyaml llama-cpp-python
```

For GPU acceleration on the inference machine, install llama-cpp-python with the appropriate backend:

```bash
# NVIDIA (CUDA)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# Intel Arc / integrated GPU (Vulkan)
CMAKE_ARGS="-DGGML_VULKAN=on" pip install llama-cpp-python

# CPU only (no extra args needed)
pip install llama-cpp-python
```

> On Windows, prebuilt wheels are available at [abetlen/llama-cpp-python releases](https://github.com/abetlen/llama-cpp-python/releases) if you want to avoid compiling.

Transfer `data/model/dinner.gguf` and `config.yaml` to the inference machine, then:

```bash
python bot/bot.py
```

The bot will load the model and start listening. It replies randomly based on `reply_chance` in `config.yaml` (default 0.01%).

---

## data/ layout

```
data/
  raw/          # scraped channel JSONs (one per channel)
  processed/    # train.jsonl + val.jsonl
  model/
    lora_adapter/   # saved after training
    dinner.gguf     # final inference model
```

`data/` is gitignored — transfer it manually between machines.

---

## Notes

- The Discord bot account used for scraping needs access to all channels you want to include
- More messages from the target user = better impersonation — aim for 10k+ examples
- `reply_chance` in config controls how often the bot speaks; tune to taste
- The bot sends replies using Discord's reply feature so it's clear which message it's responding to
