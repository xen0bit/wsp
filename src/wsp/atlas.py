"""Lever atlas: scale one component at a time and record how the identity answer moves.

For every (component, gain) trial we record, over the plain-prompt identity items:
  p_man, p_woman, p_ai   mean probability of each option on the 3-way questions
  lo_man, lo_woman, lo_ai mean log-odds of each option (more sensitive than probabilities)
  p_person               mean P(person) on the person-vs-AI question
  guard_kl               mean KL from baseline on the guard panels (collateral damage)
Deltas against the unedited baseline are what the atlas reports.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from wsp.panel.metrics import kl
from wsp.panel.run import PanelResult

OPTS3 = ("man", "woman", "ai_model")
EPS = 1e-9


def identity_stats(items, res: PanelResult) -> dict:
    p3 = {o: [] for o in OPTS3}
    person = []
    for it in items:
        if it.kind != "target":
            continue
        p = res.probs[it.id]
        opts = it.options()
        if set(opts) == set(OPTS3):
            for o in OPTS3:
                p3[o].append(float(p[opts.index(o)]))
        elif "person" in opts:
            person.append(float(p[opts.index("person")]))
    out = {}
    for o, key in zip(OPTS3, ("man", "woman", "ai")):
        arr = np.clip(np.array(p3[o]), EPS, 1 - EPS)
        out[f"p_{key}"] = float(arr.mean())
        out[f"lo_{key}"] = float(np.mean(np.log(arr) - np.log1p(-arr)))
    out["p_person"] = float(np.mean(person)) if person else float("nan")
    return out


def guard_kl(items, res: PanelResult, base: PanelResult) -> float:
    vals = [kl(base.probs[it.id], res.probs[it.id]) for it in items if it.kind == "guard"]
    return float(np.mean(vals)) if vals else 0.0


def sweep(ev, comps: list[str], scales: list[float], out: Path, log=print) -> None:
    """Append one JSON line per trial to ``out``; already-recorded trials are skipped."""
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["comp"], r["scale"]))
    base = ev.run({})
    ev.base_res = base
    b = identity_stats(ev.items, base)
    with open(out, "a") as f:
        if ("base", 1.0) not in done:
            f.write(json.dumps({"comp": "base", "scale": 1.0, **b, "guard_kl": 0.0}) + "\n")
        n = 0
        t0 = time.time()
        for comp in comps:
            for s in scales:
                if (comp, s) in done:
                    continue
                res = ev.run({comp: s})
                row = {"comp": comp, "scale": s, **identity_stats(ev.items, res), "guard_kl": guard_kl(ev.items, res, base)}
                f.write(json.dumps(row) + "\n")
                f.flush()
                n += 1
                if n % 25 == 0:
                    log(f"{n} trials, {(time.time() - t0) / n:.2f}s/trial, last {comp}@{s}: "
                        f"lo_man {row['lo_man'] - b['lo_man']:+.2f} lo_ai {row['lo_ai'] - b['lo_ai']:+.2f} kl {row['guard_kl']:.3f}")


def load_atlas(path: Path) -> tuple[dict, list[dict]]:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    base = next(r for r in rows if r["comp"] == "base")
    trials = [r for r in rows if r["comp"] != "base"]
    for r in trials:
        for k in ("man", "woman", "ai"):
            r[f"d_{k}"] = r[f"lo_{k}"] - base[f"lo_{k}"]
    return base, trials
