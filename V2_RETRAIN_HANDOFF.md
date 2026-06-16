# v2 Retrain Handoff — Diagnosis & Fix List

> Written 2026-06-15 after deploying the v2 GGUF on the ASUS laptop (Arc 140T, Vulkan)
> and discovering the model behaves poorly. This documents everything found so the
> training machine (RTX 4070 Ti) can fix the data pipeline and retrain from scratch.

## TL;DR

The bot **runs** (laptop inference stack is fully working), but the model's behavior is
bad: it overwhelmingly picks `none`, and when `none` is forcibly disabled it picks `react`
far more than `reply` and **never** picks `gif`. Root cause is **mostly in the training
data pipeline (`scraper/process_v2.py`)**, compounded by **train/inference format
mismatches in `bot/bot.py`**. There is also a **GGUF conversion gotcha** to avoid repeating.

Retrain is warranted. Fix the data bugs below, re-run `process_v2.py → train.py →
export.py`, reconvert the GGUF, then re-transfer to the laptop.

### Status: code fixes already applied (2026-06-15)
These were implemented on the laptop branch and are ready — just pull them onto the
training machine and re-run the pipeline:
- ✅ **#1 none double-weighting** — fixed in `process_v2.py` (none appended once).
- ✅ **#2 react cap + gif oversample** — `react_cap_ratio` knob added; `gif_oversample`
  bumped to 10 in config. (Finer per-action ratio targeting still optional.)
- ✅ **#4 role-mention cleaning** — `clean_content` now strips `<@&id>`.
- ✅ **#3 format mismatches** — RESOLVED 2026-06-15. Decisions: (a) Dinner's own context
  messages are labeled `you:` in training (`build_context_lines` now takes `dinner_id`),
  matching bot.py; (b) **stop merging** consecutive same-author messages — `merge_consecutive`
  removed from `process_v2.py` (bot.py already doesn't merge); (c) action-mix knobs kept as-is.
  Also aligned `bot.py`'s context cleaning to training: new `clean_context()` strips role
  pings, resolves user mentions to **display names** (was usernames), resolves channel
  mentions, normalizes custom emoji, replaces URLs with `[link]`, collapses whitespace, and
  tags attachments (`[+attachment]`/`[shared media]`) — previously none of this happened at
  inference. System-prompt `/no_think` suffix on GGUF left as-is (harmless; grammar forces JSON).
- New config knobs (`config.example.yaml` / `config.yaml`): `gif_oversample: 10`,
  `react_cap_ratio: 0.4`. **Make sure the training machine's `config.yaml` has these.**

---

## What currently works (don't re-debug these)

- **Laptop inference stack**: `llama-cpp-python==0.3.29` (Vulkan) installs & runs on
  Python 3.14 (the prebuilt wheel is `py3-none-win_amd64`, ABI-agnostic — Python version
  is NOT a constraint). Vulkan sees the Arc 140T (16GB) and offloads all 33 layers.
  Install cmd: `pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/vulkan --prefer-binary`
- **The re-converted GGUF loads** (`data/model/dinner_q8.gguf`, n_layer=32) and generates
  in-character text. The earlier broken file is fixed (see GGUF section).
- `config.yaml`, `gif_index.json`, discord.py — all present and correct on the laptop.

---

## Issues, in priority order

### 1. `none` is massively over-weighted in training data  ✅ IMPLEMENTED
**File:** `scraper/process_v2.py`, lines ~349–371.

`target_none` is computed against `action_count`, which is **already recency-weighted**
(reply/react/gif lists each appended `multiplier` times). Then each sampled `none`
candidate is recency-multiplied **again**:

```python
action_count = len(reply_pairs) + len(react_pairs) + len(reply_react_pairs) + len(gif_pairs)  # already weighted
target_none = int(action_count * none_ratio / (1 - none_ratio))   # = ~0.54 * weighted action count
sampled_nones = random.sample(none_candidates, target_none)
for ...:
    multiplier = recency_multiplier(...)
    for _ in range(multiplier):        # <-- DOUBLE weighting
        none_pairs.append(...)
```

Net effect: configured `none_ratio: 0.35` actually trains at roughly **50–60% `none`**.
The model learns that staying silent is the most likely action. (Check the
"Action distribution (after weighting)" printout from the last `process_v2.py` run to
confirm the real %.)

**Fix (simplest, exact):** append each sampled `none` **once** (drop the
`for _ in range(multiplier)` loop for none). Then `none_pairs == target_none` and the
final ratio equals the configured `none_ratio`. If recency weighting of `none` is
desired, instead divide `target_none` by the average multiplier — but append-once is
cleaner and correct.

### 2. Action mix is skewed toward low-effort actions (react ≫ reply, gif ≈ 0)  ✅ PARTLY IMPLEMENTED
**Files:** `scraper/process_v2.py` pair-generation; `config.yaml` training knobs.

Empirical inference measurements (forced, i.e. `none` disabled in grammar):
- Clean direct @mention question → ~100% `reply`.
- Real chaotic group-chat context (the `#pew` paste) → **~42% `react`, 58% `reply`**.
- `gif` → **0%** in every test.

Why:
- **react over-represented**: every message Dinner reacted to becomes a `react` pair, and
  reactions are very common in chat. Combined with the `none` bias (#1), the model defaults
  to minimal engagement (`none`/`react`) and treats `react` as the "soft none".
- **gif starved**: even with `gif_oversample: 4`, gif pairs are a tiny fraction; the
  model's `gif` prior is ~0. It essentially never fires.

**Fixes to consider for the retrain:**
- Cap or downweight `react`-only pairs so `reply` dominates engagement (e.g. target an
  explicit ratio like reply:react:gif:reply_react = 55:20:10:15 of the non-none pairs).
- Raise `gif_oversample` substantially (e.g. 8–12) or set an explicit gif floor.
- Print the post-weighting action distribution and tune to a deliberate target rather
  than letting raw counts decide.

### 3. Train/inference format mismatches  ★ FIX IN bot.py (align to training)
The model is sampled at inference on inputs formatted **differently** from training,
pushing it off-distribution (worsens #1/#2). Training format is set by
`build_context_lines`/`make_entry` in `process_v2.py`; inference format by `build_prompt`
in `bot/bot.py` (~line 164).

| Aspect | Training (`process_v2.py`) | Inference (`bot.py`) | Action |
|---|---|---|---|
| URLs in context | replaced with `[link]` (`clean_context_content`, line 77) | **left raw** (`build_prompt` does no URL sub) | Add URL→`[link]` in `build_prompt`. (Seen live: a raw github URL in `#pew` context.) |
| Consecutive same-author msgs | **merged** into one line (`merge_consecutive`) | not merged | Merge in `build_prompt`, or stop merging in training. Pick one. |
| Bot's own messages | labeled with real `author_name` | labeled `you:` (bot.py line 169) | Mismatch. `CLAUDE.md` claims `you:` labeling but `process_v2.py` does NOT do it. Decide one convention and apply to BOTH. |
| Speaker names | scraped `author_name` (numeric nicks in this server) | `display_name` | Align to the same field; if nicks changed since scrape they drift. |
| System prompt | exact `SYSTEM_PROMPT` | `SYSTEM_PROMPT + " /no_think"` for GGUF | Harmless (grammar forces JSON) but a deviation; consider matching. |

### 4. Role mentions never cleaned → model emits raw `<@&id>`  ✅ IMPLEMENTED
**File:** `scraper/process_v2.py`, `clean_content` (lines 60–72).

`clean_content` strips user mentions (`<@id>`), channel mentions (`<#id>`), and custom
emoji, but **not role mentions** (`<@&roleid>`). So role pings pass through verbatim into
training text, and the model learned to output raw `<@&123...>` — which would **ping that
role live** at inference. Add a `<@&(\d+)>` → `@rolename` (or drop) rule.

### 5. Numeric-nickname mentions (working-ish, note for context)
Users in this server have **numeric nicknames** (`17`, `41`, `48`...). The model outputs
mentions like `@18` — this is correct (it's tagging real people). `bot.py`
`resolve_output_mentions` resolves `@<nick>` → real mention via `guild.members`, but:
- depends on the **Server Members intent** + populated member cache (else left as literal text);
- the `mentions` JSON field is **dead code** (the function ignores its `mention_names` param
  and only regex-replaces inline `@name` in `text`).

Not blocking, but if mentions show as literal `@18` in Discord, the member cache/intent is
the cause.

---

## Already fixed on the laptop (carry these fixes forward / they're in bot.py now)

- **GGUF conversion missing NextN layer** — the *first* exported GGUF declared
  `qwen35.block_count=33` + `nextn_predict_layers=1` but omitted the NextN/MTP layer's
  tensors (`blk.32.nextn.eh_proj.weight`, etc.), so llama.cpp aborted with
  `missing tensor 'blk.32.attn_norm.weight'`. **Re-converting fixed it.** When you reconvert
  after retrain: ensure the convert step either includes the NextN layer tensors or excludes
  that layer cleanly (block_count=32, no nextn). Verify with a quick load before transferring.
- **UnicodeEncodeError on emoji** (`bot/bot.py`) — cp1252 Windows console crashed when
  `print`-ing emoji-containing model output, dropping those messages before dispatch.
  Fixed by reconfiguring stdout/stderr to UTF-8 at startup.
- **Forced replies could still be silent** (`bot/bot.py`) — when `@mentioned`/replied-to,
  the model could still vote `none`. Now `build_action_grammar(guild, allow_none=False)`
  drops `none` from the GBNF grammar for forced calls. NOTE: this is what exposed the
  react-vs-reply skew (#2) — disabling `none` shifts mass to `react`. Consider, for
  `@mention` specifically, also excluding bare `react` (allow only reply / reply_react /
  gif) so a direct address always yields text.
- **`config.yaml`**: `reply_on_mention` set `true` (was `false`, so @mentions never forced);
  `model_path` corrected to `dinner_q8.gguf`. `engagement_chance` is currently `1` for
  testing — reset to ~`0.05` for normal use (at `1` the bot evaluates/reacts to every message).

---

## Suggested retrain workflow (training machine)

1. **Fix `process_v2.py`**: (a) none double-weighting (#1), (b) role-mention cleaning (#4),
   (c) action-mix targets + higher gif oversample (#2).
2. **Decide the canonical context format** and make `process_v2.py` and `bot.py` match
   exactly (#3): URL→`[link]`, consecutive-merge, own-message label, name field.
3. Re-run `process_v2.py`; **inspect the printed action distribution** — confirm none ≈ your
   target and react isn't dominating reply.
4. `train.py` → `export.py` → reconvert GGUF (verify it loads, see GGUF note).
5. Transfer `dinner_q8.gguf` to the laptop (overwrite). No laptop-side changes needed.

## Quick reference — empirical behavior (current model, laptop)

| Condition | reply | react | gif | none |
|---|---|---|---|---|
| Direct @mention question, forced | ~100% | 0% | 0% | — |
| Real `#pew` group chat, forced (numeric names) | 58% | 42% | 0% | — |
| Same, word names | 67% | 33% | 0% | — |
| Direct @mention, normal (none allowed) | 77% | 0% | 0% | 23% |
| Mundane undirected, normal | ~60% | 0% | 0% | ~40% |

(Custom guild emojis were tested and ruled out as a cause of the react skew.)
