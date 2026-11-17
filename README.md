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
| identity | target | the assistant's self-report ("what are you?") under plain and persona system prompts; target = AI |
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
