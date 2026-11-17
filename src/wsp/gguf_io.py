"""GGUF access at the byte level.

Every tensor is viewed as a matrix of ``n_rows`` rows of ``ne0`` elements (ggml order:
``ne0`` is the contiguous row length, i.e. the *input* dimension of a matmul weight).
Rows are split into quantization blocks; each block carries one or two fp16 scales
that multiply every value in the block. Editing those scales gives exact (up to fp16
rounding) multiplicative edits of quantized weights without re-quantizing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType as QT
from gguf import GGUFReader
from gguf.constants import GGML_QUANT_SIZES
from gguf.quants import dequantize

# byte offsets (inside one block) of the fp16 scales that multiply the whole block.
# value = d * f(q) [- dmin * g(...)]: scaling all listed fields by s scales every value by s.
BLOCK_SCALE_FIELDS: dict[QT, tuple[int, ...]] = {
    QT.Q4_0: (0,),        # d, qs[16]
    QT.Q4_1: (0, 2),      # d, m, qs[16]
    QT.Q5_0: (0,),        # d, qh[4], qs[16]
    QT.Q5_1: (0, 2),      # d, m, qh[4], qs[16]
    QT.Q8_0: (0,),        # d, qs[32]
    QT.Q2_K: (80, 82),    # scales[16], qs[64], d, dmin
    QT.Q3_K: (108,),      # hmask[32], qs[64], scales[12], d
    QT.Q4_K: (0, 2),      # d, dmin, scales[12], qs[128]
    QT.Q5_K: (0, 2),      # d, dmin, scales[12], qh[32], qs[128]
    QT.Q6_K: (208,),      # ql[128], qh[64], scales[16], d
}
FLOAT_TYPES = {QT.F32: np.float32, QT.F16: np.float16}

FP16_MAX = 65504.0


@dataclass(frozen=True)
class TensorInfo:
    name: str
    qtype: QT
    shape: tuple[int, ...]  # ggml order: (ne0, ne1, ...)
    offset: int             # absolute byte offset of the data in the file
    n_bytes: int

    @property
    def ne0(self) -> int:
        return self.shape[0]

    @property
    def n_rows(self) -> int:
        return int(np.prod(self.shape[1:])) if len(self.shape) > 1 else 1

    @property
    def block_size(self) -> int:
        return GGML_QUANT_SIZES[self.qtype][0]

    @property
    def type_size(self) -> int:
        return GGML_QUANT_SIZES[self.qtype][1]

    @property
    def blocks_per_row(self) -> int:
        return self.ne0 // self.block_size

    @property
    def row_bytes(self) -> int:
        return self.blocks_per_row * self.type_size

    def row_range(self, row: int) -> tuple[int, int]:
        """Absolute byte range [start, end) of one row."""
        start = self.offset + row * self.row_bytes
        return start, start + self.row_bytes


class GGUFModel:
    """Read-only view of a GGUF file plus helpers to read/write raw tensor bytes."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.reader = GGUFReader(self.path, "r")
        self.tensors: dict[str, TensorInfo] = {
            t.name: TensorInfo(
                name=t.name,
                qtype=t.tensor_type,
                shape=tuple(int(x) for x in t.shape),
                offset=int(t.data_offset),
                n_bytes=int(t.n_bytes),
            )
            for t in self.reader.tensors
        }

    # metadata -----------------------------------------------------------------
    def kv(self, key: str, default=None):
        f = self.reader.fields.get(key)
        if f is None:
            return default
        try:
            return f.contents()
        except Exception:  # noqa: BLE001 - some fields have no python representation
            return default

    @property
    def arch(self) -> str:
        return self.kv("general.architecture", "unknown")

    def arch_kv(self, key: str, default=None):
        return self.kv(f"{self.arch}.{key}", default)

    # raw bytes ----------------------------------------------------------------
    def raw(self, name: str) -> np.ndarray:
        """uint8 view (no copy) of a tensor's bytes, shape (n_rows, row_bytes)."""
        t = self.tensors[name]
        buf = np.memmap(self.path, dtype=np.uint8, mode="r", offset=t.offset, shape=(t.n_bytes,))
        return buf.reshape(t.n_rows, t.row_bytes)

    def dequant(self, name: str, rows: slice | np.ndarray | None = None) -> np.ndarray:
        """float32 matrix (n_rows, ne0) or a subset of rows."""
        t = self.tensors[name]
        raw = self.raw(name)
        if rows is not None:
            raw = raw[rows]
        if t.qtype in FLOAT_TYPES:
            return np.ascontiguousarray(raw).view(FLOAT_TYPES[t.qtype]).astype(np.float32).reshape(-1, t.ne0)
        return dequantize(np.ascontiguousarray(raw), t.qtype).astype(np.float32).reshape(-1, t.ne0)

    def sha256(self) -> str:
        return file_sha256(self.path)


def file_sha256(path: str | Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def bytes_sha256(b: bytes | np.ndarray) -> str:
    return hashlib.sha256(bytes(b)).hexdigest()


# block-scale codec ------------------------------------------------------------

def scale_blocks(raw_rows: np.ndarray, qtype: QT, type_size: int, blocks: np.ndarray | slice | None, scale: float | np.ndarray) -> np.ndarray:
    """Return a copy of ``raw_rows`` (uint8, (n, row_bytes)) with the selected blocks scaled.

    ``scale`` is a scalar or an array broadcastable to (n_rows, n_selected_blocks).
    Values become ``scale * old`` up to fp16 rounding of the block scale.
    """
    if qtype in FLOAT_TYPES:
        raise TypeError("float tensors are edited element-wise, not via block scales")
    fields = BLOCK_SCALE_FIELDS.get(qtype)
    if fields is None:
        raise NotImplementedError(f"block scaling not supported for {qtype.name}")
    out = np.array(raw_rows, copy=True)
    n, row_bytes = out.shape
    view = out.reshape(n, row_bytes // type_size, type_size)
    sel = view if blocks is None else view[:, blocks]
    s = np.asarray(scale, dtype=np.float32)
    for off in fields:
        d = sel[..., off:off + 2].copy().view(np.float16)[..., 0].astype(np.float32)
        new = np.clip(d * s, -FP16_MAX, FP16_MAX).astype(np.float16)
        sel[..., off:off + 2] = new[..., None].view(np.uint8)
    if blocks is not None:
        view[:, blocks] = sel
    return out


def scale_float(raw_rows: np.ndarray, qtype: QT, cols: np.ndarray | slice | None, scale: float | np.ndarray) -> np.ndarray:
    """Element-wise scaling of F32/F16 rows (e.g. norm weights)."""
    dt = FLOAT_TYPES[qtype]
    vals = np.array(raw_rows, copy=True).view(dt).astype(np.float32)
    if cols is None:
        vals *= scale
    else:
        vals[:, cols] *= scale
    return vals.astype(dt).view(np.uint8)
