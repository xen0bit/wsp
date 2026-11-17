"""Run a panel against a /v1/systemone server and record option probabilities."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from wsp.panel import Item, state_key
from wsp.server import SystemOneClient


@dataclass
class PanelResult:
    """Option probabilities per item, in ``Item.options()`` order."""

    probs: dict[str, np.ndarray]
    meta: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "meta": self.meta,
            "usage": self.usage,
            "probs": {k: [float(x) for x in v] for k, v in self.probs.items()},
        }

    @classmethod
    def from_json(cls, d: dict) -> "PanelResult":
        return cls(probs={k: np.asarray(v, dtype=np.float64) for k, v in d["probs"].items()}, meta=d.get("meta", {}), usage=d.get("usage", {}))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "PanelResult":
        return cls.from_json(json.loads(Path(path).read_text()))


def group_by_state(items: list[Item]) -> list[list[Item]]:
    """Items that share a state go in one request (the state is prefilled once)."""
    groups: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        groups[state_key(it.state)].append(it)
    return list(groups.values())


def _probs(item: Item, answer: dict) -> np.ndarray:
    if answer["type"] == "noul":
        p = float(answer["noul"])
        return np.array([p, 1.0 - p])
    pr = answer["probabilities"]
    return np.array([float(pr[o]) for o in item.options()])


def run_panel(client: SystemOneClient, items: list[Item], meta: dict | None = None) -> PanelResult:
    t0 = time.perf_counter()
    probs: dict[str, np.ndarray] = {}
    n_in = n_out = 0
    for group in group_by_state(items):
        resp = client.ask(group[0].state, {it.id: it.question for it in group})
        for it in group:
            probs[it.id] = _probs(it, resp["answers"][it.id])
        n_in += resp["usage"]["input_tokens"]
        n_out += resp["usage"]["output_tokens"]
    usage = {"input_tokens": n_in, "passes": n_out, "seconds": time.perf_counter() - t0, "requests": len(group_by_state(items))}
    return PanelResult(probs=probs, meta=meta or {}, usage=usage)
