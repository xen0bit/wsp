"""Architecture component maps: named model parts -> overlay edits that scale them.

A component's edits multiply its contribution to the residual stream by ``s`` exactly
(s=0 ablates it). Names:

  L{i}.mixer           attention / linear-attention (DeltaNet) output of layer i
  L{i}.mlp             MLP output of layer i
  L{i}.head{h}         one attention head (or the smallest block-aligned head group)
  L{i}.n{a}-{b}        MLP neurons a..b-1 (rows of ffn_up; the SwiGLU output is linear in up)
"""

from __future__ import annotations

import math
import re
from wsp.gguf_io import FLOAT_TYPES, GGUFModel
from wsp.overlay import ScaleEdit


class ArchMap:
    def __init__(self, model: GGUFModel):
        self.m = model
        self.arch = model.arch
        self.n_layers = int(model.arch_kv("block_count"))
        if self.arch not in ("qwen35", "gemma4"):
            raise NotImplementedError(f"no component map for arch {self.arch}")

    # tensors ---------------------------------------------------------------
    def has(self, name: str) -> bool:
        return name in self.m.tensors

    def mixer_out(self, i: int) -> tuple[str, str]:
        if self.has(f"blk.{i}.attn_output.weight"):
            return f"blk.{i}.attn_output.weight", "attn"
        if self.has(f"blk.{i}.ssm_out.weight"):
            return f"blk.{i}.ssm_out.weight", "linear"
        raise KeyError(f"layer {i}: no mixer output tensor")

    def n_heads(self, i: int, mixer_type: str) -> int:
        if mixer_type == "attn":
            return int(self.m.arch_kv("attention.head_count"))
        return int(self.m.arch_kv("ssm.time_step_rank"))  # DeltaNet value heads

    # components --------------------------------------------------------------
    def mixer(self, i: int, s: float) -> list[ScaleEdit]:
        if self.arch == "gemma4":
            return [ScaleEdit(tensor=f"blk.{i}.post_attention_norm.weight", scale=s)]
        name, _ = self.mixer_out(i)
        return [ScaleEdit(tensor=name, scale=s)]

    def mlp(self, i: int, s: float) -> list[ScaleEdit]:
        if self.arch == "gemma4":
            return [ScaleEdit(tensor=f"blk.{i}.post_ffw_norm.weight", scale=s)]
        return [ScaleEdit(tensor=f"blk.{i}.ffn_down.weight", scale=s)]

    def head_groups(self, i: int) -> list[tuple[int, int]]:
        """Head index ranges [a, b) that can be scaled independently (block-aligned)."""
        name, mt = self.mixer_out(i)
        t = self.m.tensors[name]
        nh = self.n_heads(i, mt)
        hd = t.ne0 // nh
        bs = 1 if t.qtype in FLOAT_TYPES else t.block_size
        g = max(1, math.lcm(hd, bs) // hd)
        return [(a, min(a + g, nh)) for a in range(0, nh, g)]

    def heads(self, i: int, a: int, b: int, s: float) -> list[ScaleEdit]:
        name, mt = self.mixer_out(i)
        t = self.m.tensors[name]
        hd = t.ne0 // self.n_heads(i, mt)
        if t.qtype in FLOAT_TYPES:
            return [ScaleEdit(tensor=name, cols=f"{a * hd}:{b * hd}", scale=s)]
        bs = t.block_size
        assert (a * hd) % bs == 0 and (b * hd) % bs == 0, "head group not block aligned"
        return [ScaleEdit(tensor=name, blocks=f"{a * hd // bs}:{b * hd // bs}", scale=s)]

    def neurons(self, i: int, a: int, b: int, s: float) -> list[ScaleEdit]:
        return [ScaleEdit(tensor=f"blk.{i}.ffn_up.weight", rows=f"{a}:{b}", scale=s)]

    def n_neurons(self, i: int) -> int:
        return self.m.tensors[f"blk.{i}.ffn_up.weight"].n_rows

    # names -------------------------------------------------------------------
    def edits_for(self, comp: str, s: float) -> list[ScaleEdit]:
        m = re.fullmatch(r"L(\d+)\.(mixer|mlp|head(\d+)(?:-(\d+))?|n(\d+)-(\d+))", comp)
        if not m:
            raise ValueError(f"bad component name {comp!r}")
        i = int(m.group(1))
        part = m.group(2)
        if part == "mixer":
            return self.mixer(i, s)
        if part == "mlp":
            return self.mlp(i, s)
        if part.startswith("head"):
            a = int(m.group(3))
            b = int(m.group(4)) if m.group(4) else a + 1
            return self.heads(i, a, b, s)
        return self.neurons(i, int(m.group(5)), int(m.group(6)), s)

    def layer_components(self) -> list[str]:
        return [f"L{i}.{k}" for i in range(self.n_layers) for k in ("mixer", "mlp")]

    def head_components(self) -> list[str]:
        out = []
        for i in range(self.n_layers):
            for a, b in self.head_groups(i):
                out.append(f"L{i}.head{a}" if b == a + 1 else f"L{i}.head{a}-{b}")
        return out

    def mixer_type(self, i: int) -> str:
        return self.mixer_out(i)[1]
