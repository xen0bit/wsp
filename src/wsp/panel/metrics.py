"""Panel metrics: accuracy, NLL, Brier, ECE, target margins, and drift from a baseline."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from wsp.panel import Item
from wsp.panel.run import PanelResult

EPS = 1e-9
MARGIN_CAP = 6.0  # nats; a saturated target item stops pulling on the objective


def item_stats(item: Item, p: np.ndarray) -> dict:
    opts = item.options()
    t = opts.index(item.answer)
    pt = float(np.clip(p[t], EPS, 1 - EPS))
    onehot = np.zeros_like(p)
    onehot[t] = 1.0
    return {
        "p_target": pt,
        "correct": float(int(np.argmax(p)) == t),
        "nll": -np.log(pt),
        "brier": float(np.sum((p - onehot) ** 2)),
        "margin": float(np.log(pt) - np.log(1 - pt)),  # log-odds of the target option
        "p_top": float(np.max(p)),
    }


def ece(p_top: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p_top, edges) - 1, 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p_top[m].mean() - correct[m].mean())
    return float(total)


def reliability(p_top: np.ndarray, correct: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p_top, edges) - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.any():
            out.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(m.sum()), "conf": float(p_top[m].mean()), "acc": float(correct[m].mean())})
    return out


def kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, EPS, 1)
    q = np.clip(q, EPS, 1)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def summarize(items: list[Item], res: PanelResult, base: PanelResult | None = None, by=("suite", "split")) -> dict[str, dict]:
    """Aggregate metrics per (suite, split) group plus an 'all' row per kind."""
    rows: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        if it.id not in res.probs:
            continue
        s = item_stats(it, res.probs[it.id])
        if base is not None and it.id in base.probs:
            s["kl"] = kl(base.probs[it.id], res.probs[it.id])
            s["flip"] = float(np.argmax(base.probs[it.id]) != np.argmax(res.probs[it.id]))
        key = "/".join(str(getattr(it, b)) for b in by)
        rows[key].append(s)
        rows[f"{it.kind}/*"].append(s)
    out = {}
    for key, ss in sorted(rows.items()):
        arr = {k: np.array([s[k] for s in ss]) for k in ss[0]}
        out[key] = {
            "n": len(ss),
            "acc": float(arr["correct"].mean()),
            "nll": float(arr["nll"].mean()),
            "brier": float(arr["brier"].mean()),
            "ece": ece(arr["p_top"], arr["correct"]),
            "p_target_mean": float(arr["p_target"].mean()),
            "p_target_min": float(arr["p_target"].min()),
            "margin_mean": float(arr["margin"].mean()),
            "margin_capped": float(np.minimum(arr["margin"], MARGIN_CAP).mean()),
            "margin_min": float(arr["margin"].min()),
        }
        if "kl" in arr:
            out[key]["kl_mean"] = float(arr["kl"].mean())
            out[key]["kl_max"] = float(arr["kl"].max())
            out[key]["flips"] = int(arr["flip"].sum())
    return out


def objective(items: list[Item], res: PanelResult, base: PanelResult, lam: float = 10.0, split: str = "tune") -> dict:
    """J = capped target margin (mean) - lam * guard KL (mean). Higher is better."""
    tgt, guard = [], []
    for it in items:
        if it.split != split or it.id not in res.probs:
            continue
        if it.kind == "target":
            tgt.append(min(item_stats(it, res.probs[it.id])["margin"], MARGIN_CAP))
        else:
            guard.append(kl(base.probs[it.id], res.probs[it.id]))
    tm = float(np.mean(tgt)) if tgt else 0.0
    gk = float(np.mean(guard)) if guard else 0.0
    return {"J": tm - lam * gk, "target_margin": tm, "target_margin_min": float(np.min(tgt)) if tgt else 0.0, "guard_kl": gk}
