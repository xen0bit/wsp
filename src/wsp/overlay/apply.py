"""Compile overlays to byte patches and apply them to GGUF files.

A patch is ``{tensor_name: new_raw_bytes (n_rows, row_bytes) uint8}``. Every op keeps the
tensor's type and byte size, so patches are written in place: into a copy of the base
(``apply_to_file``) or into a long-lived working copy that is reverted after each trial
(``WorkingCopy``). Only rows that actually changed are written.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np

from wsp.gguf_io import FLOAT_TYPES, GGUFModel, bytes_sha256, scale_blocks, scale_float
from wsp.overlay import DirEdit, Overlay, ScaleEdit, parse_index
from wsp.quant import quantize_rows

Patch = dict[str, np.ndarray]


def compile_overlay(model: GGUFModel, ov: Overlay, root: Path | None = None) -> Patch:
    """Apply the edits in order to in-memory copies of the touched tensors."""
    patch: Patch = {}
    for e in ov.edits:
        t = model.tensors.get(e.tensor)
        if t is None:
            raise KeyError(f"no tensor {e.tensor!r} in {model.path.name}")
        cur = patch.get(e.tensor)
        if cur is None:
            cur = np.array(model.raw(e.tensor))
        if isinstance(e, ScaleEdit):
            rows = parse_index(e.rows, t.n_rows)
            sub = cur if rows is None else cur[rows]
            if t.qtype in FLOAT_TYPES:
                if e.blocks is not None:
                    raise ValueError(f"{e.tensor} is {t.qtype.name}: use cols, not blocks")
                new = scale_float(sub, t.qtype, parse_index(e.cols, t.ne0), e.scale)
            else:
                if e.cols is not None:
                    raise ValueError(f"{e.tensor} is {t.qtype.name}: use blocks (of {t.block_size}), not cols")
                new = scale_blocks(sub, t.qtype, t.type_size, parse_index(e.blocks, t.blocks_per_row), e.scale)
            if rows is None:
                cur = new
            else:
                cur[rows] = new
        elif isinstance(e, DirEdit):
            u = e.u.load(root)
            if u.shape != (t.n_rows,):
                raise ValueError(f"{e.tensor}: u has shape {u.shape}, expected ({t.n_rows},)")
            u = u / np.linalg.norm(u)
            w = _dequant_raw(cur, t)
            w = w + (e.alpha - 1.0) * np.outer(u, u @ w)
            cur = _quant_raw(w, t)
        else:  # pragma: no cover
            raise TypeError(e)
        patch[e.tensor] = cur
    return patch


def _dequant_raw(raw: np.ndarray, t) -> np.ndarray:
    if t.qtype in FLOAT_TYPES:
        return raw.view(FLOAT_TYPES[t.qtype]).astype(np.float32).reshape(t.n_rows, t.ne0)
    from gguf.quants import dequantize

    return dequantize(raw, t.qtype).astype(np.float32).reshape(t.n_rows, t.ne0)


def _quant_raw(w: np.ndarray, t) -> np.ndarray:
    if t.qtype in FLOAT_TYPES:
        return w.astype(FLOAT_TYPES[t.qtype]).view(np.uint8).reshape(t.n_rows, t.row_bytes)
    return quantize_rows(w, t.qtype)


def changed_rows(model: GGUFModel, patch: Patch) -> dict[str, np.ndarray]:
    return {name: np.flatnonzero((np.asarray(model.raw(name)) != new).any(axis=1)) for name, new in patch.items()}


def write_patch(path: Path, model: GGUFModel, patch: Patch, rows: dict[str, np.ndarray] | None = None) -> int:
    """Write changed rows into ``path`` (same layout as ``model``). Returns bytes written."""
    rows = rows or changed_rows(model, patch)
    n = 0
    with open(path, "r+b") as f:
        for name, new in patch.items():
            t = model.tensors[name]
            for r in rows[name]:
                f.seek(t.offset + int(r) * t.row_bytes)
                f.write(new[r].tobytes())
                n += t.row_bytes
    return n


def tensor_hashes(model: GGUFModel, patch: Patch) -> dict[str, dict]:
    out = {}
    for name, new in patch.items():
        rows = np.flatnonzero((np.asarray(model.raw(name)) != new).any(axis=1))
        out[name] = {
            "base_sha256": bytes_sha256(np.asarray(model.raw(name))),
            "edited_sha256": bytes_sha256(new),
            "changed_rows": int(rows.size),
        }
    return out


def apply_to_file(base: Path, ov: Overlay, out: Path, root: Path | None = None) -> Overlay:
    """Write ``out`` = base + overlay. Returns the overlay with ``result`` filled in."""
    model = GGUFModel(base)
    patch = compile_overlay(model, ov, root)
    shutil.copyfile(base, out)
    written = write_patch(out, model, patch)
    res = ov.model_copy(deep=True)
    res.result = {"tensors": tensor_hashes(model, patch), "bytes_written": written}
    return res


class WorkingCopy:
    """A copy of the base model (default in /dev/shm) that trials patch and revert in place."""

    def __init__(self, base: Path, workdir: Path = Path("/dev/shm/wsp")):
        self.base = GGUFModel(base)
        workdir.mkdir(parents=True, exist_ok=True)
        self.path = workdir / base.name
        self.marker = self.path.with_suffix(".dirty")
        if self.marker.exists() or not self.path.exists() or self.path.stat().st_size != base.stat().st_size:
            tmp = self.path.with_suffix(".tmp")
            shutil.copyfile(base, tmp)
            os.replace(tmp, self.path)
            self.marker.unlink(missing_ok=True)
        self._applied: dict[str, np.ndarray] = {}

    def apply(self, ov: Overlay, root: Path | None = None) -> int:
        self.revert()
        patch = compile_overlay(self.base, ov, root)
        rows = changed_rows(self.base, patch)
        self.marker.touch()
        self._applied = rows
        return write_patch(self.path, self.base, patch, rows)

    def apply_patch(self, patch: Patch) -> int:
        self.revert()
        rows = changed_rows(self.base, patch)
        self.marker.touch()
        self._applied = rows
        return write_patch(self.path, self.base, patch, rows)

    def revert(self) -> None:
        if not self._applied:
            return
        orig = {name: np.asarray(self.base.raw(name)) for name in self._applied}
        write_patch(self.path, self.base, orig, self._applied)
        self._applied = {}
        self.marker.unlink(missing_ok=True)

    def verify_clean(self) -> bool:
        """Byte-compare touched-or-not: full file hash against the base (slow, ~seconds)."""
        from wsp.gguf_io import file_sha256

        return file_sha256(self.path) == file_sha256(self.base.path)
