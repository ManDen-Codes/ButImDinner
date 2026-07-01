# ButImDinner

Discord bot that impersonates a friend ("Dinner"/`dinnerlore`) by fine-tuning local LLMs on their
Discord message history. v2 uses **two trained models run together** ("cluster" in one bot process):
a **persona model** that generates plain message text (with inline gifs, @mentions, and emoji), and a
tiny dedicated **reaction model** that decides which emoji (if any) to react with. The bot does *not*
output structured JSON — the model just writes Dinner's message text; gifs/reactions/silence are
handled around it.

> **v1→v2 pivot:** the original v2 used a single model emitting an action-JSON (`reply`/`react`/`gif`/
> `reply_react`/`none`). That failed in deployment (never gif'd, over-reacted, over-silenced). Root
> cause was architectural: baking the *"should I speak?"* decision into generation. The rebuild
> (documented in `PERSONA_FINETUNING_RESEARCH.md`) moves to plain-text generation + an external
> engagement gate, matching what working chat-clones actually do.

## Architecture

```
scraper → data processing → QLoRA/LoRA fine-tuning (x2) → GGUF export (x2) → discord bot (2-model local inference)
```

**Stack**: Python, discord.py, pure HuggingFace (transformers + peft + trl + bitsandbytes), llama-cpp-python (inference)
**Persona base model**: `Qwen/Qwen3-8B-Base` — **base, not instruct** (avoids the `<think>`/chat-template
bug class that plagued v1-actionsystem); QLoRA 4-bit. Qwen3 chosen over Mistral/Llama/Gemma because the
training stack is already validated on it (Gemma's 256k vocab also inflates the peft fp32 embed upcast).
**Reaction base model**: `Qwen/Qwen3-0.6B-Base` — tiny context→emoji/`none` classifier, plain bf16 LoRA
(small enough to skip 4-bit quant), output GBNF-constrained to Dinner's emoji vocab + `none`.
**Training machine**: RTX 4070 Ti (12GB VRAM); training runs in WSL2 (`~/butimdinner`, native Linux fs)
**Inference machine**: ASUS laptop — Intel Core Ultra 285H, Intel Arc 140T iGPU, 32GB RAM — runs GGUF via llama-cpp-python (Vulkan backend); or any machine with the merged HF model

## Project Structure

- `config.example.yaml` — template config (copy to `config.yaml` and fill in)
- `scraper/scrape.py` — connects as Discord bot, exports all messages from a server to `data/raw/` (includes reaction metadata)
- `scraper/scrape_reactions.py` — second-pass scraper that fetches reaction user IDs for messages with reactions
- `scraper/process.py` — v1 processor (legacy): plain-text ChatML pairs
- `scraper/process_v2.py` — v2 processor: builds the plain-text persona dataset **and** the reaction dataset, plus `gif_index.json` and `reaction_emoji_freq.json`
- `training/train.py` — persona QLoRA fine-tune (Qwen3-8B-Base) using pure HuggingFace stack
- `training/train_reactions.py` — reaction model LoRA fine-tune (Qwen3-0.6B-Base)
- `training/eval.py` — standalone eval script (runs on a checkpoint without OOMing)
- `training/export.py` — `python export.py [persona|reaction]`: merges the LoRA into its base, prints the GGUF conversion command
- `bot/bot.py` — Discord bot; loads persona (required) + reaction (optional) models, auto-detects backend (HF if path is a directory, GGUF if `.gguf` file)

## Workflow

1. `python scraper/scrape.py` — scrape messages (needs `config.yaml` with bot token, guild ID, Dinner's user ID)
2. `python scraper/scrape_reactions.py` — fetch reaction user IDs (second pass, only messages with reactions)
3. `python scraper/process_v2.py` — build both training datasets + `gif_index.json` + `reaction_emoji_freq.json`
4. `python training/train.py` — fine-tune the persona model (GPU machine); resumes automatically from latest checkpoint
5. `python training/train_reactions.py` — fine-tune the reaction model (fast)
6. `python training/export.py persona` and `python training/export.py reaction` — merge → `data/model/merged` + `reaction_merged`; then convert each to GGUF Q8_0 with llama.cpp
7. `python bot/bot.py` — run the bot

## Current Status

- v1 training complete (Qwen3 4B, 5 epochs, ~29k pairs — plain text replies only)
- v1-actionsystem ("original v2") complete but **abandoned** — the action-JSON approach; adapters archived (`lora_adapter_v1_actionsys_*`)
- **v2 rebuild complete** (June–July 2026):
  - persona: Qwen3-8B-Base, QLoRA, 3 epochs (~55k plain-text pairs), final `train_loss` ~1.05
  - reaction: Qwen3-0.6B-Base, LoRA, best-eval checkpoint kept (epoch 2, `eval_loss` ~0.51; `load_best_model_at_end`)
  - both exported to GGUF Q8_0: `dinner_q8.gguf` (8.7GB) + `dinner_reaction_q8.gguf` (639MB)
  - smoke-tested locally (persona voice + format clean, no `<think>` leakage; reaction vocab valid, no class collapse)
  - deploying to ASUS laptop

## Key Details

- `data/` is gitignored — transfer it between machines manually
- **Two-model system, plain text** — the persona model emits Dinner's message text directly (no JSON/action labels). Gifs are inline tokens, mentions/emoji are inline; the reaction model is a separate call.
- **Base (non-instruct) models** — deliberately: kills the `<think>`/chat-template/`enable_thinking` bug class that cost days in v1-actionsystem. No chat template applied anywhere; prompt = raw transcript.
- **Training data format** — TRL prompt/completion pairs with `completion_only_loss=True` (loss masked to Dinner's turn only). Persona rows: `{"prompt": "<transcript>\ndinnerlore:", "completion": " <text>"}`. Reaction rows: `{"prompt": "<transcript>\n[reaction]:", "completion": " :emoji:" | " none"}`.
- **Transcript format** — `name: text` lines; consecutive same-author messages within `merge_window_minutes` merged into one turn (natural multi-line bursts); conversation blocks split at `conversation_gap_minutes` gaps; context capped to `context_max_turns` / `context_max_chars`; bot's own turns labeled with `persona_name`.
- **Inline gifs (symmetric)** — Tenor posts render as `[gif: <slug words>]` in **both** input context and output targets (extracted from Tenor URL slugs). At inference the bot detects `[gif:]` in output and resolves it against Dinner's own GIF history (`data/processed/gif_index.json`); no external API. Toggle via `gif_enabled`; natural rate ~2.3% with a `gif_oversample` knob.
- **Reactions** — separate Qwen3-0.6B-Base model (`context → :emoji:`/`none`), GBNF-constrained to Dinner's emoji vocab. Data is sparse (~1.3% of messages) and imbalanced (`:tomfoolery:` dominates positives): `none` downsampled 1:1 (`reaction_none_ratio`), vocab thresholded (`reaction_min_emoji_count`, `reaction_top_k_emoji`). **Overfits fast** — `train_reactions.py` uses `load_best_model_at_end` (eval_loss bottoms ~epoch 2 then climbs). Phase-1 fallback: leave `reaction_model_path: ""` to use a weighted emoji-frequency heuristic (`reaction_emoji_freq.json`) behind the same interface.
- **Engagement gate (external, NOT the model)** — always engages when someone replies to the bot's message (`reply_chain_decay` tapering); optionally always on @mention (`reply_on_mention`); otherwise rolls `engagement_chance` in `allowed_channels`. The model never owns the silence decision. The reaction step fires on `react_chance` (or when engaged).
- **Stateless inference** — no session memory; each response is built from a fresh transcript of recent channel messages. Fine-tune = voice; RAG (not built) would be facts.
- **Bot backend auto-detection** — per model path: directory → HF transformers; `.gguf` → llama-cpp-python.
- **Import order**: `import torch` must be LAST import in the training scripts or `datasets` crashes silently on Windows.
- **Mention handling** — input `@mentions` resolved to readable names; output `@name` matched against guild members (username + nickname) → real Discord mention; `@everyone`/`@here` pings suppressed.
- **Data processing filters** — URL-only / attachment-only / too-short (`min_reply_chars`) Dinner replies dropped (gif-only targets kept); URLs in context replaced with `[link]`; recency weighting duplicates recent messages (`recency_weights`); dominant-speaker downsample above `dominant_speaker_cap_ratio` (anti-overfit; not triggered — Dinner talks broadly).
- **Scrape cutoff date** — `scraper.cutoff_date` excludes messages after a date (keep bot-generated messages out of training data post-deployment).
- **Discord intents** — requires `message_content` and `members` intents in the Discord Developer Portal.
- Config is in `config.yaml` (gitignored) — copy from `config.example.yaml`. Sections: `processing` / `training` / `reaction` / `bot`.
- **`persona_name` must match** the username for `dinner_user_id` in `user_index.json` (what `process_v2` trained the speaker label under).
- Zero-cost solution: all open-source, runs fully local, no API calls.

## Training Dependencies (GPU machine)

```
pip install torch --index-url https://download.pytorch.org/whl/cu130 --force-reinstall
pip install datasets "transformers==4.57.6" peft trl==0.24.0 bitsandbytes pyyaml
pip uninstall torchvision -y
```

Note: `trl` must be pinned to `0.24.0` — newer versions have a Windows Unicode bug reading Jinja templates.
`SFTConfig` here uses `max_length` (not `max_seq_length`), `completion_only_loss`, and `packing`.

**CRITICAL — pin `transformers==4.57.6`:** transformers 5.x (e.g. 5.12.1) causes a ~20x QLoRA
slowdown on this stack (Qwen3-8B went 5.6s/step → 123s/step; GPU pegged at 100% but glacial).
Do NOT `pip install -U transformers` — and note that upgrading transformers also silently pulls a
CPU-only torch, so if you ever must upgrade, reinstall the cu130 torch afterward.

**Training quirks on 12GB VRAM (persona):**
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set in the scripts (harmless no-op on torch 2.12 Windows)
- Do NOT use `gradient_checkpointing=True` — ~10x slower here
- Persona uses `eval_strategy="no"` (eval OOMs at 12GB; use `training/eval.py` separately). The tiny reaction model *can* eval inline (`eval_strategy="epoch"`).
- `save_total_limit` keeps a few recent checkpoints to manage disk space
- ~3 epochs of persona ≈ 13h on the 4070 Ti (~6.2s/step, ~10.3k steps); reaction model ≈ 6–12 min

**WSL2 note:** training runs from `~/butimdinner` on the WSL native filesystem (fast); the git repo lives
on the Windows side. WSL tears down the distro when the launching `wsl.exe` exits — launch long training
in its own terminal window (`Start-Process wsl.exe … <launcher>.sh`) so it survives; scripts auto-resume
from the latest checkpoint.

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

1. Transfer from the GPU machine into `data/`:
   - `data/model/dinner_q8.gguf` (persona)
   - `data/model/dinner_reaction_q8.gguf` (reaction)
   - `data/processed/gif_index.json`
   - `data/processed/reaction_emoji_freq.json`
2. Transfer `config.yaml` (gitignored)
3. Clone repo + install dependencies (above)
4. In `config.yaml`: `model_path: "data/model/dinner_q8.gguf"` and
   `reaction_model_path: "data/model/dinner_reaction_q8.gguf"` (or `""` to use the emoji heuristic)
5. `python bot/bot.py`
