# AGENTS.md — working in `wsp` (Weight Space Programming on GGUF)

`wsp` edits a quantized model's weights directly to change what it says about itself
("what are you?", "where are you?", …). Edits are small JSON **overlays** of component gains,
found by sweeping the model with cheap, deterministic **decision-model requests** (TypeSafe/Jev
`/v1/systemone`: one forward pass per answer, no generation) and applied byte-for-byte to the
GGUF. The result is an ordinary GGUF that runs in stock llama.cpp.

Read this whole file before changing panels, axes, or the search loop.

---

## 1. Setup

### llama.cpp fork (xen0bit/llama.cpp)

The fork adds `POST /v1/systemone` (PR #1) and `--wsp-patch` /
`POST /wsp/tensors` (PR #3: write weight bytes into a running server). `wsp` needs both; both are on `master`.

```sh
cd ~/Projects/llama.cpp && git checkout master && git pull
cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build build -j
```

* CUDA 12.4 (apt `nvidia-cuda-toolkit`) does not support gcc 15 → host compiler `g++-13`.
* The prebuilt web-UI bundle fails its asset check → UI off (not needed).
* cmake/ninja can come from `uv tool install cmake ninja`.
* Missing system tools: ask the user before installing (they have sudo); don't work around it.
* Paths are configurable: `WSP_LLAMA_CPP`, `WSP_LLAMA_BIN`, `WSP_MODELS`, `WSP_RUNS`.

### This repo

```sh
cd ~/Projects/wsp && uv sync          # gguf-py comes from the fork (editable path dependency)
uv run pytest -q                       # ~2 min; needs the models below
```

### Pick a model

Models are untracked files in `models/` (git-ignored). Source: `/run/media/remy/MODELS/models/lmstudio/models`.

| model | why |
|---|---|
| `Qwen3.5-4B-Q4_K_M.gguf` (lmstudio-community) | default; fast iteration. Hybrid: Gated DeltaNet + full attention every 4th layer, 32 layers |
| `Qwen3.5-9B-Q4_K_M.gguf` | same arch, bigger |
| `gemma-4-12B-it-QAT-Q4_0.gguf` (lmstudio-community, **not** the HauhauCS one) | best decision-model quality in the fork's benchmarks; 48 layers, SWA |

```sh
cp /run/media/remy/MODELS/models/lmstudio/models/lmstudio-community/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf models/
uv run wsp gguf info Qwen3.5-4B-Q4_K_M.gguf
uv run wsp gguf tensors -g blk.3.        # ggml order: shape[0]=ne0 is the row length (matmul input)
```

Component maps exist for `qwen35` and `gemma4` (`src/wsp/arch/__init__.py`). A new architecture
needs a map there (see §6).

---

## 2. Establish a baseline

```sh
uv run wsp panel build --openjev path/to/openjev/benchmarks/data/authored144.jsonl
uv run wsp panel calibrate -m Qwen3.5-4B-Q4_K_M.gguf
```

`calibrate` runs the full panel set (774 items) twice from cold starts and once with
`--systemone-permute`, and writes `runs/baseline-<model>/{a,b,perm,baseline}.json` + `report.json`:
accuracy, NLL, Brier, ECE, reliability bins, and order-bias flips.

Qwen3.5-4B baseline: bit-identical across restarts; guard accuracy 0.85, ECE 0.05;
openjev authored144 0.80 (0.84 permuted); strong option-order bias (127 flips) — which is why
every axis question is asked in both option orders.

**Determinism rules** (all enforced by `ServerConfig`): `-np 1`, fixed `-b/-ub`, `-cram 0`,
sequential requests. Parallel slots or concurrent requests change results (measured up to 0.1).

---

## 3. How it works

```
axes + guards ──► decision panel ──► one trial = patch weights → run panel → revert
                                      │
                    atlas: every component × gain, its effect on every option + guard KL
                                      │
                    compose: stack most efficient levers under the budgets ──► overlay JSON
                                      │
                    wsp lib apply / wsp gui ──► patched GGUF ──► llama-cli / llama-server
```

### Decision panels
`/v1/systemone` puts the `state` in the system message and the question in the user message,
labels options `A`, `B`, …, and softmaxes the next-token logits over the labels. One prefill per
question, no decode. A request bundles all questions that share a state.

### Weight edits (`src/wsp/gguf_io.py`, `src/wsp/overlay/`)
* **`scale`** on quantized tensors multiplies the fp16 per-block scales (`d`, and `dmin` for
  K-quants): exact up to fp16 rounding, **no re-quantization**, same bytes size.
  Float tensors (norms) are scaled element-wise.
* **`dir`** (`W' = W + (α−1)·u uᵀW`) dequantizes, edits, and re-quantizes into the same type via
  ggml's C quantizer (rel. error ~2–6e-4). Re-quantization noise limits how cleanly a direction
  can be removed (~20% left for a random direction on Q6_K).
* Components (`ArchMap.edits_for`): `L{i}.mixer` (attention or DeltaNet output), `L{i}.mlp`
  (`ffn_down`), `L{i}.head{h}` / `L{i}.head{a}-{b}` (block-aligned head groups), `L{i}.n{a}-{b}`
  (rows of `ffn_up`; SwiGLU output is linear in `up`). On gemma4 the mixer/MLP gains use the F32
  post-norms (`layer_output_scale` scales the whole residual, don't use it).

### Trials (`src/wsp/search.py`)
`HotEvaluator` starts `llama-server --wsp-patch --load-mode none --slot-save-path …`, writes the
changed rows (chunked ≤16 MB per request), erases the slot cache, runs the panel (optionally
greedy generation too), and writes the original bytes back. Verified **bit-identical** to writing
a GGUF and reloading. ~2.5–3.6 s per trial depending on panel size; the panel, not the patching,
is the cost. `Evaluator` (reload from a `/dev/shm` working copy) is the slower reference path; its
working copy carries a `.dirty` marker so an interrupted run can't leave it silently patched.

### Scores
* **Target**: mean probability and mean log-odds of each option over the axis items
  (both option orders, tune split). Atlas deltas are in log-odds.
* **Guard KL**: mean KL(base‖edited) of option probabilities on guard items (basic facts,
  counterfactual stories, 16 openjev items). Collateral drift. Default budget **0.15**.
* **Gen sim**: greedy 32-token outputs on 4 neutral prompts vs the base model (difflib ratio).
  Default floor **0.6**. Needed because the one-token panel cannot see repetition loops or broken
  stopping — an overlay at guard KL 0.44 generated "2020202…" before this check existed.

---

## 4. Curating good Jev requests

Load the `typesafe:typesafe-ai` skill when writing questions. The rules that mattered here:

* **Human test**: a person shown only the state and question should answer at a glance.
  State = what the model would see (`system_prompt`, `user_message`), nothing else.
* **One judgment per question**; plain wording; no preambles or output-format instructions.
* **Descriptive option names** (`far_future`, not `B`) — names are sent to the model.
  Criteria describe each option as a concrete situation ("It says it is in Tokyo, Japan.").
* **Ask every choice in both option orders** (`fwd`/`rev`). Small models have strong position bias.
* **Keep a default option** the base model picks (`ai_model`, `present`, `on_servers`,
  `help_users`); targets are the other options. Without it, "nothing fits" gets forced onto a target.
* **Split tune / holdout by prompt**: held-out personas and messages never appear in tuning.
* **Guards must be hard enough.** Easy guards (capitals, story characters) missed damage that
  openjev authored144 caught (accuracy 0.80 → 0.57). Keep a slice of hard judgment items in the
  search guard and verify on the full panel.
* **Avoid prompt gymnastics**: don't add criteria clauses to rescue a failing item or tune
  thresholds to a handful of cases. Fix the state or split the question.

Inspect exactly what is sent:

```python
from wsp.axes import AXES, axis_items
from wsp.panel.run import group_by_state
items = axis_items(AXES["place"], "tune")
g = group_by_state(items)[0]
print({"model": "jev-latest", "state": g[0].state, "questions": {i.id: i.question for i in g}})
```

---

## 5. The library

```
library/
  who|what|when|where|why/<model-stem>/<axis>-<option>.json   one overlay per axis value
  mix/<model-stem>/<a>+<b>[+<c>].json                         combinations
  atlas/<model-stem>/model.json                               base file, sha256, arch
  atlas/<model-stem>/<axis-set>.{layers,fine}.jsonl           raw sweep data (one row per trial)
```

Every overlay file is self-describing: base sha256/arch, the edits, `meta.levers`
(component → gain), and `meta.score` (all axes, guard KL, gen sim). The catalog is a directory scan.

| axis | category | options (default last) |
|---|---|---|
| `sex` | who | male, female, ai_model |
| `species` | what | cat, dog, ai_model |
| `era` | when | medieval, far_future, present |
| `place` | where | paris, tokyo, on_servers |
| `purpose` | why | make_art, explore, help_users |

```sh
uv run wsp lib build --axes era,place,purpose   # baseline → layer sweep → fine sweep → compose (~15 min on 4B)
uv run wsp lib compose era-far_future           # re-compose from existing atlas (no sweep)
uv run wsp lib levers                           # ranked levers per option, per axis set
uv run wsp lib mix --axes sex,species           # all cross-axis combinations
uv run wsp lib mix sex-male species-dog         # one combination
uv run wsp lib list                             # catalog with scores
uv run wsp lib apply sex-male species-dog       # → models/<stem>.sex-male+species-dog.gguf
uv run wsp gui                                  # pick a GGUF, compatibility per overlay, tick, apply
```

Mixing takes the **union** of levers: a lever two overlays share is applied once; conflicting
gains on one component multiply (logged). Compatibility (`check_compat`): `exact` = sha256 match;
`compatible` = arch and every edited tensor's type/shape match (scores were measured on the base
and may differ); otherwise `incompatible`.

Use the result with normal inference:

```sh
~/Projects/llama.cpp/build/bin/llama-cli -m "models/Qwen3.5-4B-Q4_K_M.sex-male+species-dog.gguf" -ngl 99 -cnv
```

---

## 6. Adding things

### A new overlay (new axis or option)
1. Add an `Axis` to `AXES` in `src/wsp/axes.py`: `category` (who/what/when/where/why), `options`
   (name → criterion), `default` (what the base model says), and if the axis isn't about
   self-description, its own `question` and 8 `messages` (5 tune, 3 holdout, one non-English).
2. Check the baseline: the default should dominate and targets should be low but non-zero.
   If a target is ~0 everywhere, rephrase the criterion (human test) before sweeping.
3. `uv run wsp lib build --axes <name>` (add `--fine-mixers/--fine-mlps/--candidates` to trade time).
4. `uv run wsp lib levers`, `uv run wsp lib list`; apply and chat-test the GGUF.
5. Verify on the full panel when it matters: `uv run wsp overlay verify <overlay.json> -b runs/baseline-*/baseline.json`.
6. Commit the overlay JSON and the atlas files.

### A new model
Copy the GGUF to `models/`, run `wsp panel calibrate -m …`, then `wsp lib build -m … --axes …`.
A new architecture needs `ArchMap` support: which tensor is each layer's mixer output, the MLP
down/up tensors, head count/width for head groups.

### A new guard
Add a builder in `src/wsp/panel/suites.py` (`BUILDERS`), `wsp panel build`, and include it in
`lib.guard_items()` if it should constrain the search. Re-run `calibrate`.

---

## 7. Status and known limits (Qwen3.5-4B)

* Single overlays reach target P ≈ 0.2–0.4 (from 0.001–0.09) at guard KL ≤ 0.15, gen sim ≥ 0.6.
  Strongest: `era-far_future` (0.09 → 0.41, ahead of medieval), `sex-male`, `species-cat`.
  Weakest: `purpose-explore` (0.001 → 0.085, make_art stays ahead), `sex-female`.
* Most levers are **anti-default** (they lower `ai_model`/`present`/`on_servers` and raise every
  other option together) rather than option-specific. Selective levers exist but are rarer
  (e.g. `L19.head9` ablated: female +0.8, male unchanged).
* Free generation still mostly answers with the default (e.g. "I am an AI"); the decision-mode
  shift is the measured effect.
* 2-way sex+species mixes are borderline (guard KL 0.20–0.27, gen sim 0.58–0.75). 3-way when+where+why mixes stack 8–12 levers, reach
  guard KL 0.2–1.8 and gen sim 0.44–0.68, and win at most 2 of 3 axes: **not usable as-is**.

Next steps, in order of expected value: rank levers by *selectivity* (Δtarget − Δrival on the same
axis); jointly compose mixes under one budget instead of unioning separate overlays; intermediate
gains; wider fine sweeps (more layers); per-neuron levers inside the best 256-blocks.

---

## 8. Conventions and gotchas

* `models/` and `runs/` are git-ignored; `library/` and `panels/` are committed.
* Never `pkill -f llama-server` from a shell whose own command line contains that string — it
  kills the shell. Kill by PID (`ps -eo pid,args | awk '$2 ~ /llama-server$/ {print $1}'`).
* Slot actions (`/slots/0?action=erase`) need `--slot-save-path`.
* `--no-mmap` is spelled `--load-mode none` in this llama.cpp; needed so CPU-side tensors are writable.
* Keep sweeps coarse-to-fine: layers (64 × gains) first, then heads/neurons only in the layers that
  moved something. Don't launch hour-long exhaustive sweeps for a first pass.
* Commit messages end with the `Co-Authored-By` line from the session's attribution instructions.
