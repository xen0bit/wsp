"""Trial evaluators: apply component gains, run a decision panel, revert.

Each trial: compile overlay -> patch the /dev/shm working copy -> restart llama-server
(~1.3 s) -> run a small panel -> revert. Deterministic end to end.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from wsp.arch import ArchMap
from wsp.gguf_io import GGUFModel
from wsp.overlay import BaseRef, Overlay
from wsp.overlay.apply import WorkingCopy, compile_overlay
from wsp.panel import Item
from wsp.panel.metrics import MARGIN_CAP, item_stats, kl
from wsp.panel.run import PanelResult, run_panel
from wsp.server import LlamaServer, ServerConfig, SystemOneClient

class Evaluator:
    def __init__(self, base: Path, items: list[Item], port: int, lam: float, cap: float = MARGIN_CAP):
        self.model = GGUFModel(base)
        self.arch = ArchMap(self.model)
        self.items = items
        self.wc = WorkingCopy(base)
        self.srv = LlamaServer(ServerConfig(model=self.wc.path, port=port))
        self.lam = lam
        self.cap = cap
        self.base_res: PanelResult | None = None
        self.n_evals = 0

    def overlay(self, gains: dict[str, float], meta: dict | None = None) -> Overlay:
        edits = []
        for comp, s in sorted(gains.items()):
            for e in self.arch.edits_for(comp, s):
                edits.append(e.model_copy(update={"note": comp}))
        return Overlay(base=BaseRef(file=self.model.path.name, arch=self.model.arch), meta=meta or {}, edits=edits)

    def run(self, gains: dict[str, float]) -> PanelResult:
        self.wc.apply_patch(compile_overlay(self.model, self.overlay(gains)))
        try:
            self.srv.restart()
            res = run_panel(SystemOneClient(self.srv.url), self.items)
        finally:
            self.wc.revert()
        self.n_evals += 1
        return res

    def score(self, res: PanelResult) -> dict:
        tgt, guard, pt = [], [], []
        for it in self.items:
            if it.kind == "target":
                s = item_stats(it, res.probs[it.id])
                tgt.append(min(s["margin"], self.cap))
                pt.append(s["p_target"])
            else:
                guard.append(kl(self.base_res.probs[it.id], res.probs[it.id]))
        g = float(np.mean(guard)) if guard else 0.0
        t = float(np.mean(tgt))
        return {"J": t - self.lam * g, "margin": t, "margin_min": float(np.min(tgt)),
                "p_min": float(np.min(pt)), "p_mean": float(np.mean(pt)), "guard_kl": g}

    def close(self) -> None:
        self.srv.stop()
        self.wc.revert()


def _encode_writes(model: GGUFModel, rows_by_tensor: dict[str, np.ndarray], data: dict[str, np.ndarray],
                   max_bytes: int = 16 << 20) -> list[bytes]:
    """Bodies for POST /wsp/tensors (each <= max_bytes); contiguous rows are coalesced into records."""
    import struct

    bodies: list[bytes] = []
    out = bytearray()
    for name, rows in rows_by_tensor.items():
        if rows.size == 0:
            continue
        t = model.tensors[name]
        runs = np.split(rows, np.flatnonzero(np.diff(rows) != 1) + 1)
        nb = name.encode()
        rows_per_rec = max(1, (max_bytes - 1024) // t.row_bytes)
        for run in runs:
            for k in range(0, run.size, rows_per_rec):
                a, b = int(run[k]), int(run[min(k + rows_per_rec, run.size) - 1]) + 1
                payload = np.ascontiguousarray(data[name][a:b]).tobytes()
                rec = struct.pack("<I", len(nb)) + nb + struct.pack("<QQ", a * t.row_bytes, len(payload)) + payload
                if out and len(out) + len(rec) > max_bytes:
                    bodies.append(bytes(out))
                    out = bytearray()
                out += rec
    if out:
        bodies.append(bytes(out))
    return bodies


class HotEvaluator(Evaluator):
    """Like Evaluator, but writes weights into the running server (--wsp-patch) instead of reloading."""

    def __init__(self, base: Path, items: list[Item], port: int, lam: float, cap: float = MARGIN_CAP):
        from wsp.overlay.apply import changed_rows  # noqa: F401

        self.model = GGUFModel(base)
        self.arch = ArchMap(self.model)
        self.items = items
        self.srv = LlamaServer(ServerConfig(model=base, port=port, no_mmap=True, extra=["--wsp-patch", "--slot-save-path", "/dev/shm/wsp-slots"]))
        self.srv.start()
        self.client = SystemOneClient(self.srv.url)
        self.lam = lam
        self.cap = cap
        self.base_res = None
        self.n_evals = 0
        self.wc = None

    def _write(self, rows, data) -> None:
        for body in _encode_writes(self.model, rows, data):
            r = self.client.http.post(f"{self.srv.url}/wsp/tensors", content=body)
            r.raise_for_status()
        r = self.client.http.post(f"{self.srv.url}/slots/0?action=erase")
        r.raise_for_status()

    def run(self, gains: dict[str, float], gen: bool = False) -> PanelResult:
        from wsp.overlay.apply import changed_rows

        patch = compile_overlay(self.model, self.overlay(gains))
        rows = changed_rows(self.model, patch)
        self._write(rows, patch)
        try:
            res = run_panel(self.client, self.items)
            if gen:
                res.meta["gen"] = generate(self.client, self.srv.url)
                self.client.http.post(f"{self.srv.url}/slots/0?action=erase").raise_for_status()
        finally:
            self._write(rows, {n: np.asarray(self.model.raw(n)) for n in rows})
        self.n_evals += 1
        return res

    def close(self) -> None:
        self.srv.stop()


GEN_PROMPTS = [
    "What is the capital of France? Answer in one sentence.",
    "List three primary colors.",
    "Translate 'good morning' into Spanish.",
    "Write one sentence about the ocean.",
]


def generate(client: SystemOneClient, url: str, prompts: list[str] = GEN_PROMPTS, n: int = 32) -> list[str]:
    """Greedy chat completions (thinking off): the generation-health probe."""
    out = []
    for p in prompts:
        r = client.http.post(f"{url}/v1/chat/completions", json={
            "messages": [{"role": "user", "content": p}], "temperature": 0, "max_tokens": n,
            "chat_template_kwargs": {"enable_thinking": False}})
        r.raise_for_status()
        out.append(r.json()["choices"][0]["message"]["content"] or "")
    return out


def gen_similarity(base: list[str], new: list[str]) -> float:
    """Mean text similarity to the unedited model's greedy outputs (1 = identical)."""
    from difflib import SequenceMatcher

    return float(np.mean([SequenceMatcher(None, a, b).ratio() for a, b in zip(base, new)]))
