# Persona Fine-Tuning: Research & Failure Analysis

Reference doc for ButImDinner. Captures what went wrong with the v2 action-system
approach, what the broader community/research does instead, and the architecture we're
pivoting to. **Read this before rebuilding** so we don't repeat the v2 saga.

---

## TL;DR

1. **Don't make the generation model decide whether to respond.** Baking a `none` action
   into the model is the documented anti-pattern and was the root cause of nearly every v2
   symptom. Separate the *"should I speak?"* decision (external gate / classifier) from the
   *"what do I say?"* generation. This is unanimous across practitioner write-ups and research.
2. **Don't use a structured action-JSON system.** No successful chat-clone does this. They
   fine-tune the model to **emit the person's message text, plain**. GIFs/reactions/silence
   are handled *outside* the model.
3. **Prefer a base (non-instruct) model** to avoid chat-template / thinking-token bugs.
4. **The model was never the problem in v2** — it learned the persona well. The architecture
   around it (action system + in-model silence decision + instruct-template inference) was.

---

## Part 1 — What we built in v2 and why it failed

### The v2 design
A single fine-tuned model outputs a JSON **action**: `reply`, `react`, `gif`, `reply_react`,
or `none`. One model call decides both *whether* and *what*. A GBNF grammar constrains output
to valid JSON. The bot dispatches based on the action.

### The symptoms in deployment
- **Never chose `gif`** (despite gif being 10.1% of training data).
- **Chose `react` far more than its 1.3% training share.**
- **Chose `none` way too much** (stayed silent / ignored interaction).

### The diagnosis (empirically verified, not guessed)
We reproduced the exact GGUF inference locally and tested the model directly. Findings:

- **The model is correct.** Given real validation contexts, it picked `gif` on
  **~75% of gif-contexts**, and handled reply/none/react appropriately. It learned the persona
  and the action mapping fine. Training loss converged cleanly (4.44 → 0.74 over 1 epoch).
- **The bug was architectural**, in two layered mistakes:
  1. **`none` baked into the model.** The model strongly prefers `none` (25% of training +
     it's the "safe" default). On *random* engagement it says `none` a lot → "too many nones."
  2. **Forced-engagement react-fallback.** When the bot is directly addressed (reply/mention),
     the code removed `none` from the grammar (`allow_none=False`). Robbed of its preferred
     "do nothing," the model fell back to a **bare `react` ~92% of the time** (verified:
     on `none`-context examples, removing `none` from the grammar produced `react` 11/12).
     That is the "reacting the most." gif was rarely the top fallback → "no gifs."

  Quick-fix applied (stopgap, not the real cure): on forced engagement also drop bare `react`,
  leaving `reply | gif | reply_react`. Verified this flips the same contexts to **reply 9/10 +
  gif** with zero react-spam. Good band-aid; the real fix is the pivot below.

### Other v2 rabbit holes (time sinks worth remembering)
- **Instruct chat-template / thinking tokens.** Qwen3's `<think>` handling, `enable_thinking`,
  `/no_think`, and llama-cpp's chat-template auto-guessing caused repeated *out-of-distribution*
  prompt bugs at inference. Multiple "fixes" (system-prompt tweak, manual ChatML) chased this
  before we proved the model itself was fine. **A base model avoids this entire class of bug.**
- **GGUF vs training format drift.** Inference prompt must match training format *exactly*;
  small drifts (a stray `/no_think`, template differences) silently degrade behavior.

### Root-cause summary
> The generation model was asked to make a decision (respond or not) that 7B-class models
> can't make reliably from inside generation. Constraining that decision at inference
> (removing `none`) just relocated the failure (→ `react`). The model was competent; the
> control architecture was wrong.

---

## Part 2 — What the community/research actually does

### Separate the engagement decision from generation (the big one)
- 7B-class models **respond at every turn** or collapse to a default when constrained — a known
  limitation of generation-only approaches. ([AI for Groups — Towards Data Science][tds])
- The clean pattern: a **binary engagement classifier / trigger model** ("should the bot reply
  to this?") gating a **generation model** that only writes the message.
  ([When to Talk — arXiv][whentotalk], engagement-classifier patents)
- Free training signal: every message the person *did* / *didn't* reply to is a labeled example
  for the engagement classifier; an `@mention` the bot ignored is a missed-engagement positive.

### Plain reply generation, not an action system
- Working clones fine-tune the model to **emit the person's message text** — no JSON actions.
  ([Watson Chua][watson], [HN: 240k messages][hn], [Cloning Discord friends][substack])
- GIFs/reactions are handled outside the model (or appear naturally as text — e.g. Tenor links
  were just text in the chat history).

### Use a base (non-instruct) model
- Watson Chua deliberately used **Mistral-7B base, not instruct**, "to avoid conflicting prompt
  templates." Instruct templates (and Qwen3's thinking tokens) fight the persona format and add
  inference fragility. ([Watson Chua][watson])

### Data formatting that works
- Group messages into **conversation blocks split by ~1h gaps**; cap block length by tokens.
- **Merge consecutive messages from the same sender within ~5 min** into one turn — this is what
  makes the clone produce natural multi-line bursts. (v2 *removed* consecutive-merge — reconsider.)
- Consistent speaker labels + a fixed template are critical even with a base model. ([Watson Chua][watson])

### Known failure modes to design against
- **Overfitting to the dominant speaker.** Watson's model overfit to his wife (75% of data) and
  hallucinated with strangers. Balance across relationships; watch our recency + multi-account skew.
- **Hallucination on novel inputs.** Mitigations: downsample dominant relationships, add
  per-relationship/persona role labels, and optionally do background-knowledge completion
  fine-tuning before chat fine-tuning. ([Watson Chua][watson])

### Fine-tuning vs RAG
- **Fine-tuning = voice/persona. RAG = facts/memory.** Cameron Tarle's group bot was largely a
  *searchable knowledge base* (RAG over chat history) for "what has everyone said." If we later
  want Dinner to *remember* facts, that's RAG layered on top — not more fine-tuning.
  ([Cameron Tarle][cameron])

---

## Part 3 — Proposed pivot architecture

```
incoming message
   │
   ├─► [Engagement gate]  (external, NOT the model)
   │     mention / reply-to-bot / reply-chain decay / engagement_chance
   │     → decides IF Dinner speaks
   │     (upgrade path: a tiny trained yes/no classifier; data labels it for free)
   │
   └─► if yes ► [Persona model]  (fine-tuned, ONE job)
                 generate Dinner's reply text in his voice — plain text, no JSON
                   │
                   └─► light post-processing:
                         • detect gif intent in output → match gif_index → send GIF
                         • reactions: drop, or a trivial separate heuristic
```

### Concrete changes vs v2
1. **Reprocess data as plain reply pairs** (context → Dinner's message text). Drop action labels.
   Add conversation grouping (~1h split) + consecutive-merge (~5 min).
2. **Fine-tune for plain reply generation**, ideally on a **base** model (kills the template/
   thinking class of bug). QLoRA is fine.
3. **GIFs stay as text** — his Tenor posts are normal training targets; at inference detect a gif
   intent and resolve via the existing `gif_index`. No competing JSON branch.
4. **Engagement stays external** (already built). Optional later upgrade: train a binary
   "did Dinner reply here?" classifier from the same scraped data.

### Why this is structurally better
- Removes the `none`/react-fallback failure by construction (the model never owns the
  silence decision).
- Removes instruct-template fragility (base model + simple fixed format).
- Matches every working real-world clone; far less novel surface area to debug.

---

## Resolved decisions (official v2 rebuild — June 2026)
Research re-verified against all six sources before building (Watson Chua's base-model
rationale, 5-min merge, 1h/3000-tok blocks, 75%-wife overfit, plain-text output all confirmed
verbatim; "separate when-from-what" confirmed via TDS + a USPTO engagement-classifier patent;
"When to Talk" arXiv 1912.09879 confirmed; HN/Edward Donner 240k confirmed, adds the "mundane,
loops" failure mode).

- **Base model:** `Qwen/Qwen3-8B-Base` — base over instruct (kills the `<think>`/chat-template
  bug class); Qwen3 family over Mistral/Llama/Gemma because the training stack is already
  validated on it (zero re-validation). Vocab/VRAM analysis: Gemma rejected (256k vocab inflates
  the peft fp32 embed upcast; gated; fussy quant).
- **Output:** plain message text. No action JSON.
- **GIFs:** inline `[gif: <slug words>]` token, **symmetric** on input context and output target
  (others' Tenor posts render the same, not `[link]`); reuses `extract_tenor_query` + `gif_index`
  + `search_gif`. Bot toggle (`gif_enabled`). Natural rate ~2.3% with a `gif_oversample` knob.
- **Reactions:** removed from the persona model. Official v2 = **two trained models** run together
  ("cluster" = both loaded in the bot): persona model + a tiny **Qwen3-0.6B-Base reaction model**
  (context→emoji/none, GBNF-constrained to Dinner's emoji vocab). Phase-1 placeholder = weighted
  emoji-frequency heuristic behind the same interface. CAVEAT: reaction data is sparse (~1.3%) +
  imbalanced (`:tomfoolery:` ~45% of positives) — `none` downsampled 1:1; watch class collapse.
- **Engagement:** external heuristic gate kept (mention / reply-to-bot / reply-chain decay /
  engagement_chance). Trained "should Dinner reply?" classifier is a later upgrade.
- **Data recipe:** conversation blocks split at >60 min; consecutive same-author merge ≤5 min;
  context cap 12 turns / 1500 chars; dominant-speaker downsample at >20% (not triggered — Dinner
  talks broadly); recency weighting retained; completion-only loss (mask context, train on his turn).

### Where each decision lives in code
- `scraper/process_v2.py` — both datasets, gif tokens, grouping/merge, balancing, emoji freq.
- `training/train.py` — Qwen3-8B-Base, prompt/completion, `completion_only_loss=True`.
- `training/train_reactions.py` — Qwen3-0.6B-Base reaction model.
- `training/export.py` — `python export.py [persona|reaction]` → merged → GGUF (Q8_0).
- `bot/bot.py` — plain-text generation, `[gif:]` segmenting, decoupled reaction step, engagement gate.
- `config.yaml` — `processing` / `training` / `reaction` / `bot` sections.

---

## Sources
- [Watson Chua — Finetuning My Clone: Training an LLM to Talk Like Me][watson]
- [HN discussion — A simulation of me: fine-tuning an LLM on 240k text messages][hn]
- [Cameron Tarle — I Built an AI Bot That Knows Everything My Friend Group Has Ever Said][cameron]
- [Cloning your Discord friends with LLMs (Substack)][substack]
- [AI for Groups: Build a Multi-User Chat Assistant Using 7B-Class Models — Towards Data Science][tds]
- [When to Talk: Chatbot Controls the Timing of Talking (arXiv)][whentotalk]

[watson]: https://medium.com/@watsonchua/finetuning-my-clone-training-an-llm-to-talk-like-me-2ee7b5ba2f88
[hn]: https://news.ycombinator.com/item?id=38847581
[cameron]: https://cameron-tarle.medium.com/i-built-an-ai-bot-that-knows-everything-my-friend-group-has-ever-said-b16e770c667e
[substack]: https://substack.com/home/post/p-140882623
[tds]: https://towardsdatascience.com/ai-for-groups-build-a-multi-user-chat-assistant-using-7b-class-models-7071ca8b4aa0/
[whentotalk]: https://arxiv.org/pdf/1912.09879
