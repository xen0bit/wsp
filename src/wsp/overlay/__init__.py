"""Weight overlays: a JSON list of edits applied to a base GGUF.

Ops
---
scale  Multiply a selection of a tensor by ``scale``.
       Quantized tensors: selection is rows x blocks, applied to the fp16 block scales
       (no re-quantization, exact up to fp16 rounding). Float tensors (norms, F32/F16):
       selection is rows x columns (elements), applied element-wise.
dir    Scale the component of a tensor's output along a unit direction ``u`` (length n_rows):
       W' = W + (alpha - 1) * u (u^T W). alpha=0 removes the direction, alpha>1 amplifies.
       The tensor is dequantized, edited, and re-quantized into its own type (same size).

Index specs (rows, blocks, cols): null/"all", an int, "a:b", "a:b,c,e:f", or a list of ints.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Annotated, Any, Literal, Union

import numpy as np
from pydantic import BaseModel, Field

IndexSpec = Union[None, int, str, list[int]]


def parse_index(spec: IndexSpec, n: int) -> np.ndarray | None:
    """None means everything."""
    if spec is None or spec == "all":
        return None
    if isinstance(spec, int):
        idx = [spec]
    elif isinstance(spec, list):
        idx = spec
    else:
        idx = []
        for part in str(spec).split(","):
            if ":" in part:
                a, b = part.split(":")
                idx.extend(range(int(a or 0), int(b or n)))
            else:
                idx.append(int(part))
    arr = np.asarray(sorted(set(idx)), dtype=np.int64)
    if arr.size and (arr[0] < 0 or arr[-1] >= n):
        raise IndexError(f"index out of range 0..{n - 1}: {spec}")
    return arr


class Vector(BaseModel):
    """A float vector, inline (base64 float32) or in a sidecar .npz (path relative to the overlay)."""

    b64: str | None = None
    npz: str | None = None
    key: str | None = None

    @classmethod
    def inline(cls, v: np.ndarray) -> "Vector":
        return cls(b64=base64.b64encode(np.asarray(v, dtype=np.float32).tobytes()).decode())

    def load(self, root: Path | None = None) -> np.ndarray:
        if self.b64 is not None:
            return np.frombuffer(base64.b64decode(self.b64), dtype=np.float32).copy()
        assert self.npz and self.key
        return np.load((root or Path(".")) / self.npz)[self.key].astype(np.float32)


class ScaleEdit(BaseModel):
    op: Literal["scale"] = "scale"
    tensor: str
    rows: IndexSpec = None
    blocks: IndexSpec = None  # quantized tensors
    cols: IndexSpec = None    # float tensors
    scale: float
    note: str | None = None


class DirEdit(BaseModel):
    op: Literal["dir"] = "dir"
    tensor: str
    u: Vector
    alpha: float
    note: str | None = None


Edit = Annotated[Union[ScaleEdit, DirEdit], Field(discriminator="op")]


class BaseRef(BaseModel):
    file: str
    sha256: str | None = None
    arch: str | None = None


class Overlay(BaseModel):
    wsp_overlay: Literal[1] = 1
    base: BaseRef
    meta: dict[str, Any] = Field(default_factory=dict)
    edits: list[Edit] = Field(default_factory=list)
    result: dict[str, Any] = Field(default_factory=dict)  # filled by apply: per-tensor hashes

    @classmethod
    def load(cls, path: str | Path) -> "Overlay":
        return cls.model_validate_json(Path(path).read_text())

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.model_dump(exclude_none=True), indent=1) + "\n")
