# ButImDinner

A Discord bot that impersonates a friend by fine-tuning a local LLM on their message history. The bot lurks in allowlisted channels and randomly replies in their style — fully local, no API costs.

## How it works

```
scraper → data processing → QLoRA fine-tuning → GGUF export → discord bot
```

- **Scraper**: logs in as a Discord bot and exports all messages from a server
- **Processor**: converts raw logs into ChatML training pairs with recency weighting and quality filtering
- **Training**: QLoRA fine-tunes Qwen3 4B on your GPU using pure HuggingFace (transformers + peft + trl)
- **Export**: merges the LoRA adapter into the base model; optionally converts to GGUF for lightweight inference
- **Bot**: auto-detects HF or GGUF backend, randomly replies in character, resolves @mentions in both input and output

## Requirements

### Hardware
- **Training**: NVIDIA GPU with 8GB+ VRAM (tested on RTX 4070 Ti 12GB)
- **Inference**: any machine with 8GB+ RAM — HF merged model or GGUF, GPU optional

### Software
- Python 3.10+
- CUDA 12.x+ (training only)

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd ButImDinner
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
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
- `bot.reply_on_mention` — always reply when @mentioned (default: `true`)
- `bot.mention_name` — name to replace bot's @mention with in model input (default: `"dinner"`)

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

Exports all messages from the server to `data/raw/` as JSON files (one per channel). The bot needs **Read Message History** permission in each channel. Scraping can be interrupted and resumed — already-scraped channels are skipped.

---

## Step 2 — Process

No new dependencies needed.

```bash
python scraper/process.py
```

Reads `data/raw/`, builds training pairs, and writes:
- `data/processed/train.jsonl` (90%)
- `data/processed/val.jsonl` (10%)

Each example is a ChatML conversation: the preceding chat context as the user turn, the target's reply as the assistant turn.

**Quality filters applied:**
- Responses containing URLs are dropped (prevents the model learning to generate fake links)
- Responses with attachments (images, GIFs) are dropped
- Responses shorter than 3 characters are dropped
- URLs in context messages are replaced with `[link]`

**Recency weighting** duplicates recent messages so the model reflects current behavior more than old messages. Configurable in `config.yaml` under `recency_weights`.

---

## Step 3 — Train

Install dependencies on the GPU machine:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130 --force-reinstall
pip install datasets transformers peft trl==0.24.0 bitsandbytes pyyaml
pip uninstall torchvision -y
```

> Adjust the cu130 index to match your CUDA version (cu124, cu126, etc.)
> Pin trl to 0.24.0 — newer versions have a Windows bug reading Jinja templates

Run training:

```bash
python training/train.py
```

Downloads Qwen3 4B (~8GB from HuggingFace, no account required), applies QLoRA 4-bit, and trains for 5 epochs. Checkpoints are saved every 150 steps — if interrupted, training resumes automatically from the latest checkpoint.

Training config in `config.yaml` under `training:`:

| Key | Default | Notes |
|-----|---------|-------|
| `base_model` | `unsloth/Qwen3-4B` | HuggingFace model ID |
| `epochs` | `5` | |
| `batch_size` | `4` | lower if OOM |
| `gradient_accumulation_steps` | `4` | effective batch = batch × accum |
| `learning_rate` | `0.0002` | |
| `lora_rank` | `64` | |
| `max_seq_length` | `512` | |

---

## Step 4 — Export

No new dependencies needed.

```bash
python training/export.py
```

Merges the LoRA adapter into the base model and saves the full merged model to `data/model/merged/`. This is usable directly by the bot.

**Optional: convert to GGUF** for lighter inference (needed for llama-cpp-python backend):

```bash
git clone https://github.com/ggerganov/llama.cpp
pip install -r llama.cpp/requirements.txt
python llama.cpp/convert_hf_to_gguf.py data/model/merged --outfile data/model/dinner_f16.gguf --outtype f16
```

Then quantize (download prebuilt llama.cpp binary from [releases](https://github.com/ggerganov/llama.cpp/releases)):

```bash
# Q8_0 — near-lossless, ~4.5GB
llama-quantize.exe data/model/dinner_f16.gguf data/model/dinner.gguf Q8_0

# Q4_K_M — smaller, ~2.5GB
llama-quantize.exe data/model/dinner_f16.gguf data/model/dinner.gguf Q4_K_M
```

---

## Step 5 — Run the bot

The bot auto-detects which backend to use based on `model_path` in `config.yaml`:
- **Directory** (e.g. `data/model/merged`) → HuggingFace transformers (requires GPU with 8GB+ VRAM)
- **`.gguf` file** (e.g. `data/model/dinner.gguf`) → llama-cpp-python

### On the training machine (HF backend)

```bash
pip install discord.py pyyaml
```

Set `model_path: "data/model/merged"` in config.yaml, then:

```bash
python bot/bot.py
```

### On a separate inference machine (GGUF + llama-cpp-python)

Install llama-cpp-python using a prebuilt wheel from [abetlen/llama-cpp-python releases](https://github.com/abetlen/llama-cpp-python/releases). Pick the wheel matching your hardware:

| Hardware | Release tag |
|----------|-------------|
| NVIDIA GPU | `cu132`, `cu126`, `cu124` etc. (match your CUDA version) |
| Intel Arc / Vulkan | `vulkan` |
| CPU only | base release (no tag) |

```bash
pip install discord.py pyyaml
pip install "https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.26-vulkan/llama_cpp_python-0.3.26-py3-none-win_amd64.whl"
```

Transfer `data/model/dinner.gguf` and `config.yaml` to the inference machine. Set `model_path: "data/model/dinner.gguf"` in config.yaml, then:

```bash
python bot/bot.py
```

---

## data/ layout

```
data/
  raw/              # scraped channel JSONs (one per channel)
  processed/        # train.jsonl + val.jsonl
  model/
    lora_adapter/   # LoRA checkpoints saved during training
    merged/         # full merged HF model (used by HF backend)
    dinner_f16.gguf # full precision GGUF (intermediate)
    dinner.gguf     # quantized GGUF (used by llama-cpp-python backend)
```

`data/` is gitignored — transfer it manually between machines.

---

## Notes

- The Discord bot account needs **Read Message History** in all channels you want scraped
- Enable **Message Content Intent** and **Server Members Intent** in the [Discord Developer Portal](https://discord.com/developers/applications) (Bot settings)
- More messages from the target user = better impersonation
- `reply_chance` in config controls how often the bot speaks (default 0.01% — raise to 5–20% for testing)
- The bot always replies when someone replies to one of its messages
- Set `reply_on_mention: true` to also always reply when @mentioned
- `mention_name` controls what the bot's @mention is replaced with in model input (default: `"dinner"`)
- If the model outputs `@name`, it's resolved against guild members and converted to a real Discord mention; `@everyone`/`@here` pings are blocked
- The bot replies using Discord's reply feature so it's clear which message triggered it
- Qwen3's built-in chain-of-thought (`<think>` tags) is disabled at both training and inference time
