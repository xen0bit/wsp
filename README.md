# wsp — Weight Space Programming on GGUF models

Program behaviour directly into a quantized model's weights, using calibrated Jev-style
decision panels (`/v1/systemone`, one forward pass per answer) as a fast, deterministic oracle.
Edits are stored as JSON **overlays** and applied byte-for-byte to a base GGUF.

## Setup

```sh
# llama.cpp fork with /v1/systemone (xen0bit/llama.cpp), CUDA build
cd ~/Projects/llama.cpp
cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF   # prebuilt UI bundle fails its asset check
cmake --build build -j

cd ~/Projects/wsp && uv sync
cp <lmstudio>/lmstudio-community/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf models/   # untracked
```

Paths can be overridden with `WSP_LLAMA_CPP`, `WSP_LLAMA_BIN`, `WSP_MODELS`, `WSP_RUNS`.

## Library workflow (per model)

```sh
uv run wsp gui                              # pick a GGUF, see which overlays fit, tick, apply
uv run wsp lib build -m Qwen3.5-4B-Q4_K_M.gguf --axes sex,species   # atlas + one overlay per axis value
uv run wsp lib compose                      # re-compose overlays from the existing atlas (no sweep)
uv run wsp lib levers                       # what raises/lowers each option (male, female, cat, dog, ai_model)
uv run wsp lib mix                          # every cross-axis combination, e.g. sex-male+species-dog
uv run wsp lib mix sex-female species-cat   # one specific mix
uv run wsp lib list                         # catalog with measured scores
uv run wsp lib apply sex-male species-dog   # -> models/Qwen3.5-4B-Q4_K_M.sex-male+species-dog.gguf
~/Projects/llama.cpp/build/bin/llama-cli -m "models/Qwen3.5-4B-Q4_K_M.sex-male+species-dog.gguf" -ngl 99 -cnv
```

Layout: `library/<who|what|when|where|why>/<model-stem>/<axis>-<opt>.json`, mixes in
`library/mix/<model-stem>/`, raw sweep data in `library/atlas/<model-stem>/`. Every overlay file is
self-describing (base sha256, levers, measured scores); the catalog is a scan of the tree.
Axes live in `src/wsp/axes.py` (`sex`: male/female/ai_model, `species`: cat/dog/ai_model); add one
there and rebuild. All axes share the same "what are you?" prompts, so one request scores every axis.

`build` runs: baseline → every layer mixer/MLP at gains 0 and 2 → heads in the 4 strongest mixer
layers and 256-neuron blocks in the strongest MLP layer (ablation) → for each axis value, stack the
most efficient levers while the guard KL stays under the budget (default 0.15) **and** greedy
generation on 4 neutral prompts stays similar to the base model (text similarity >= 0.6; the
one-token decision panel cannot see repetition loops or broken stopping). Mixes take the union of
levers: a lever two overlays share is applied once; conflicting gains multiply (with a warning). Output GGUFs keep tensor types and sizes, so they run in stock llama.cpp.

**Guard KL** = mean KL(base‖edited) of the option probabilities on guard questions (facts,
counterfactual stories, a slice of openjev authored144). It measures collateral drift; `wsp overlay
verify` re-checks an overlay on the full 774-item panel.

Trials are hot-patched: `llama-server --wsp-patch` (fork PR #3) writes the edited bytes into
the loaded weights, the slot cache is erased, the panel runs, and the original bytes are written
back. Results are bit-identical to writing a GGUF and reloading.

## Commands

```sh
wsp gguf info|tensors|row            # inspect a GGUF (ggml order: ne0 = row length)
wsp panel build --openjev authored144.jsonl
wsp panel calibrate -m Qwen3.5-4B-Q4_K_M.gguf     # baseline: 2 cold runs + permuted run
wsp panel run / compare
wsp overlay apply  OVERLAY OUT.gguf               # base + overlay -> new GGUF (+ hashes)
wsp overlay verify OVERLAY -b runs/baseline-*/baseline.json
wsp overlay diff A.gguf B.gguf
```

## Panels (`panels/*.jsonl`)

| suite | kind | what it checks |
|---|---|---|
| identity | target | (calibration) self-report under plain and persona system prompts |
| counterfactual | guard | characters in stories keep their identity (humans stay human, AI characters stay AI) |
| persona | guard | the assistant still knows its persona's job, town, employer |
| basic | guard | facts, sentiment, routing, language |
| authored144 | guard | openjev's labeled 3-option judgments (MIT, TheoLeeCJ/openjev) |

Held-out splits use personas and messages never seen in tuning.

## Overlay ops

* `scale` — quantized tensors: multiply fp16 block scales (rows × blocks); exact up to fp16
  rounding, no re-quantization. Float tensors (norms): element-wise (rows × cols).
* `dir` — `W' = W + (α−1)·u uᵀW` on the output side; re-quantized into the same type
  (relative error ~2–6e-4).

Component names (`wsp.arch`): `L{i}.mixer`, `L{i}.mlp`, `L{i}.head{h}`, `L{i}.n{a}-{b}`.
