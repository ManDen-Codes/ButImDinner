# ButImDinner

Discord bot that impersonates a friend ("Dinner") by fine-tuning a local LLM on their Discord message history. The bot randomly replies to messages in allowlisted channels.

## Architecture

```
scraper → data processing → QLoRA fine-tuning → GGUF export → discord bot (local inference)
```

**Stack**: Python, discord.py, pure HuggingFace (transformers + peft + trl + bitsandbytes), llama-cpp-python (inference)
**Base model**: Qwen3 4B (`unsloth/Qwen3-4B`, no HF license gate)
**Training machine**: RTX 4070 Ti (12GB VRAM), QLoRA 4-bit, bf16
**Inference machine**: ASUS laptop — Intel Core Ultra 285H, Intel Arc 140T iGPU, 32GB RAM — runs GGUF via llama-cpp-python (Vulkan backend); or any machine with the merged HF model

## Project Structure

- `config.example.yaml` — template config (copy to `config.yaml` and fill in)
- `scraper/scrape.py` — connects as Discord bot, exports all messages from a server to `data/raw/`
- `scraper/process.py` — converts raw messages into ChatML-style training pairs (JSONL) in `data/processed/`
- `training/train.py` — QLoRA fine-tune Qwen3 4B using pure HuggingFace stack
- `training/export.py` — merges LoRA adapter into base model, saves to `data/model/merged/`
- `bot/bot.py` — Discord bot; auto-detects backend (HF if model_path is a directory, GGUF if .gguf file)

## Workflow

1. `python scraper/scrape.py` — scrape messages (needs `config.yaml` with bot token, guild ID, Dinner's user ID)
2. `python scraper/process.py` — build training pairs from raw data
3. `python training/train.py` — fine-tune (run on GPU machine); resumes automatically from latest checkpoint
4. `python training/export.py` — merge LoRA into base model → `data/model/merged/`; optionally convert to GGUF with llama.cpp
5. `python bot/bot.py` — run the bot

## Current Status

- Training complete (5 epochs, ~29k pairs after filtering and recency weighting)
- Bot running and tested on training machine (RTX 4070 Ti) using HF backend
- Next: port to ASUS laptop using GGUF + Vulkan backend

## Key Details

- `data/` is gitignored — transfer it between machines manually
- **Unsloth was abandoned** — TRL 0.24.0 incompatibility with `<EOS_TOKEN>` placeholder; pure HF stack used instead
- **Import order**: `import torch` must be LAST import in train.py or `datasets` crashes silently on Windows
- **Thinking mode**: Qwen3 has built-in chain-of-thought (`<think>` tags); disabled via `enable_thinking=False` in both training format and inference
- **Bot backend auto-detection**: `model_path` in config — if directory → HF transformers; if `.gguf` file → llama-cpp-python
- **Data processing filters**: URL-containing responses dropped, attachment-only responses dropped, responses < 3 chars dropped; URLs in context replaced with `[link]`; recency weighting duplicates recent messages (configurable in `recency_weights` config)
- Config is in `config.yaml` (gitignored) — copy from `config.example.yaml`
- Zero-cost solution: all open-source, runs fully local, no API calls

## Training Dependencies (GPU machine)

```
pip install torch --index-url https://download.pytorch.org/whl/cu130 --force-reinstall
pip install datasets transformers peft trl==0.24.0 bitsandbytes pyyaml
pip uninstall torchvision -y
```

Note: `trl` must be pinned to `0.24.0` — newer versions have a Windows Unicode bug reading Jinja templates.

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

1. Transfer `data/model/dinner_f16.gguf` (or quantized version) and `config.yaml`
2. Clone repo / copy bot files
3. Install dependencies (above)
4. Set `model_path: "data/model/dinner.gguf"` in config.yaml (pointing to GGUF file)
5. Optionally quantize on the laptop: download llama.cpp Vulkan release, run `llama-quantize.exe dinner_f16.gguf dinner.gguf Q8_0`
6. `python bot/bot.py`
