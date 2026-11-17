"""Overlay library, organized by category then model.

library/
  <category>/<model-stem>/<axis>-<opt>.json   one overlay per axis value; category in who|what|when|where|why
  mix/<model-stem>/<a>+<b>.json                combined overlays (gains multiply on shared components)
  atlas/<model-stem>/model.json                base file, sha256, arch
  atlas/<model-stem>/<axes>.layers.jsonl       one row per (component, gain) trial: guard KL + stats of those axes
  atlas/<model-stem>/<axes>.fine.jsonl         heads / neuron blocks inside the layers that mattered

Every overlay file is self-describing (base sha256, levers, measured scores), so the catalog is
just a scan of the tree.

Pipeline (``wsp lib build``): baseline -> layer sweep -> fine sweep in the strongest layers
-> compose one overlay per axis value under a guard-KL budget -> index.
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np

from wsp.axes import Axis, axis_items
from wsp.config import REPO, panels_dir
from wsp.gguf_io import file_sha256
from wsp.overlay import BaseRef, Overlay
from wsp.panel import Item, load_panel
from wsp.panel.metrics import kl
from wsp.panel.run import PanelResult
from wsp.search import gen_similarity

EPS = 1e-9
N_OPENJEV = 16


# --- panel ------------------------------------------------------------------------------------

def guard_items() -> list[Item]:
    g = []
    for name in ("basic", "counterfactual"):
        g += load_panel(panels_dir() / f"{name}.jsonl")
    oj = [i for i in load_panel(panels_dir() / "authored144.jsonl") if i.split == "tune"]
    return g + oj[:: max(1, len(oj) // N_OPENJEV)][:N_OPENJEV]


def search_panel(axes: list[Axis], split: str = "tune") -> list[Item]:
    items = []
    for ax in axes:
        items += axis_items(ax, split)
    return items + guard_items()


def axis_stats(items: list[Item], res: PanelResult, axes: list[Axis]) -> dict:
    out = {}
    for ax in axes:
        acc = {o: [] for o in ax.options}
        for it in items:
            if it.tags.get("axis") != ax.name:
                continue
            p = res.probs[it.id]
            for o, v in zip(it.options(), p):
                acc[o].append(float(np.clip(v, EPS, 1 - EPS)))
        out[ax.name] = {o: {"p": float(np.mean(v)), "lo": float(np.mean(np.log(v) - np.log1p(-np.array(v))))} for o, v in acc.items()}
    return out


def guard_kl(items: list[Item], res: PanelResult, base: PanelResult) -> float:
    v = [kl(base.probs[i.id], res.probs[i.id]) for i in items if i.kind == "guard"]
    return float(np.mean(v)) if v else 0.0


# --- library ----------------------------------------------------------------------------------

class Library:
    def __init__(self, model_path: Path, root: Path | None = None):
        self.model_path = model_path
        self.stem = model_path.stem
        self.base_root = root or REPO / "library"
        self.atlas = self.base_root / "atlas" / self.stem
        self.atlas.mkdir(parents=True, exist_ok=True)
        meta = self.atlas / "model.json"
        if not meta.exists():
            from wsp.gguf_io import GGUFModel
            m = GGUFModel(model_path)
            meta.write_text(json.dumps({"file": model_path.name, "sha256": file_sha256(model_path), "arch": m.arch}, indent=1))
        self.meta = json.loads(meta.read_text())

    def overlay_path(self, name: str) -> Path:
        from wsp.axes import axis_of
        cat = "mix" if "+" in name else axis_of(name).category
        p = self.base_root / cat / self.stem / f"{name}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def load(self, name: str) -> Overlay:
        return Overlay.load(self.overlay_path(name))

    def entries(self) -> dict[str, dict]:
        """name -> overlay meta (levers, score, ...) for this model."""
        return {e["name"]: e for e in catalog(self.base_root) if e["model"] == self.stem}

    def make_overlay(self, ev, gains: dict[str, float], meta: dict) -> Overlay:
        ov = ev.overlay(gains, meta)
        ov.base = BaseRef(file=self.meta["file"], sha256=self.meta["sha256"], arch=self.meta["arch"])
        return ov


def catalog(root: Path | None = None) -> list[dict]:
    """Every overlay in the library: category, model, name, path, levers, score."""
    root = root or REPO / "library"
    out = []
    for p in sorted(root.glob("*/*/*.json")):
        cat, model = p.parent.parent.name, p.parent.name
        if cat == "atlas":
            continue
        ov = Overlay.load(p)
        out.append({"category": cat, "model": model, "name": p.stem, "path": p, "base": ov.base,
                    "levers": ov.meta.get("levers", {}), "score": ov.meta.get("score"), "n_edits": len(ov.edits)})
    return out


# --- sweeps -----------------------------------------------------------------------------------

def sweep(ev, axes: list[Axis], comps: list[str], scales: list[float], out: Path, log=print) -> None:
    done = set()
    if out.exists():
        done = {(r["comp"], r["scale"]) for r in map(json.loads, out.read_text().splitlines())}
    base = ev.base_res
    todo = [(c, s) for c in comps for s in scales if (c, s) not in done]
    log(f"{out.name}: {len(todo)} trials ({len(done)} already done)")
    t0 = time.time()
    with open(out, "a") as f:
        if ("base", 1.0) not in done:
            f.write(json.dumps({"comp": "base", "scale": 1.0, "guard_kl": 0.0, "axes": axis_stats(ev.items, base, axes)}) + "\n")
        for n, (c, s) in enumerate(todo, 1):
            res = ev.run({c: s})
            f.write(json.dumps({"comp": c, "scale": s, "guard_kl": guard_kl(ev.items, res, base), "axes": axis_stats(ev.items, res, axes)}) + "\n")
            f.flush()
            if n % 50 == 0 or n == len(todo):
                log(f"  {n}/{len(todo)}  {(time.time() - t0) / n:.2f}s/trial")


def atlas_files(atlas_dir: Path, axis: str | None = None) -> list[Path]:
    """Sweep files of an axis set ('sex+species.layers.jsonl'); with ``axis``, only sets containing it."""
    files = sorted(atlas_dir.glob("*.jsonl"))
    return [f for f in files if axis is None or axis in f.name.split(".")[0].split("+")]


def load_trials(paths: list[Path]) -> tuple[dict, list[dict]]:
    """Trials with per-axis log-odds deltas against their own file's baseline. Rows for the same
    (component, gain) from different axis sets are merged (axes combined, worst guard KL kept)."""
    merged: dict[tuple, dict] = {}
    base_all: dict = {"comp": "base", "scale": 1.0, "guard_kl": 0.0, "axes": {}}
    for p in paths:
        if not p.exists():
            continue
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        base = next((r for r in rows if r["comp"] == "base"), None)
        if base is None:
            continue
        base_all["axes"].update(base["axes"])
        for r in rows:
            if r["comp"] == "base":
                continue
            d = {a: {o: r["axes"][a][o]["lo"] - base["axes"][a][o]["lo"] for o in r["axes"][a]} for a in r["axes"]}
            key = (r["comp"], r["scale"])
            m = merged.setdefault(key, {"comp": r["comp"], "scale": r["scale"], "guard_kl": 0.0, "axes": {}, "d": {}})
            m["axes"].update(r["axes"])
            m["d"].update(d)
            m["guard_kl"] = max(m["guard_kl"], r["guard_kl"])
    return base_all, list(merged.values())


def rivals(axis: str, opt: str) -> list[str]:
    from wsp.axes import AXES
    return [o for o in AXES[axis].targets() if o != opt]


def selectivity(r: dict, axis: str, opt: str) -> float:
    """Δlogodds(target) minus the largest Δlogodds of another non-default option on the same axis.
    0 for a lever that lifts every option equally (anti-default); >0 only if it prefers the target."""
    d = r["d"][axis]
    rv = [d[o] for o in rivals(axis, opt)]
    return d[opt] - (max(rv) if rv else 0.0)


def lever_score(r: dict, parts: list[tuple[str, str]]) -> float:
    """Summed per-part gain (target lift + selectivity); rows missing an axis are not candidates."""
    if any(a not in r["d"] for a, _ in parts):
        return float("-inf")
    return sum(r["d"][a][o] + selectivity(r, a, o) for a, o in parts)


def objective(stats: dict, parts: list[tuple[str, str]]) -> float:
    """sum over parts of lo(target) + [lo(target) - max lo(rival)]."""
    total = 0.0
    for a, o in parts:
        lo = stats[a][o]["lo"]
        rv = [stats[a][x]["lo"] for x in rivals(a, o)]
        total += lo + (lo - (max(rv) if rv else lo))
    return total


def strongest_layers(trials: list[dict], kind: str, n: int) -> list[int]:
    """Layers whose `kind` (mixer|mlp) component moved any non-AI option the most."""
    score: dict[int, float] = {}
    for r in trials:
        if not r["comp"].endswith("." + kind):
            continue
        layer = int(r["comp"][1:].split(".")[0])
        from wsp.axes import AXES
        m = max(abs(v) for a, opts in r["d"].items() for o, v in opts.items() if o != AXES[a].default)
        score[layer] = max(score.get(layer, 0.0), m)
    return [l for l, _ in sorted(score.items(), key=lambda x: -x[1])[:n]]


# --- compose / mix / evaluate -----------------------------------------------------------------

MIN_GEN_SIM = 0.6


def compose(ev, trials: list[dict], parts: list[tuple[str, str]], budget: float, candidates: int, log=print,
            min_gen_sim: float = MIN_GEN_SIM) -> tuple[dict, dict]:
    """Greedy joint composition for one or more (axis, option) targets.

    Candidates are ranked by summed lift + selectivity per unit of guard KL; a lever is kept only if
    the measured stack improves the joint objective, stays under the guard-KL budget, and keeps
    greedy generation close to the base model."""
    scored = [(lever_score(r, parts), r) for r in trials if r["guard_kl"] <= budget]
    scored = [(sc, r) for sc, r in scored if sc > 0]
    ranked = [r for _, r in sorted(scored, key=lambda x: -x[0] / (x[1]["guard_kl"] + 0.02))]
    base = ev.base_res
    cur = objective(axis_stats(ev.items, base, ev.axes), parts)
    gains: dict[str, float] = {}
    best = None
    tried = 0
    for r in ranked:
        if r["comp"] in gains:
            continue
        if tried >= candidates:
            break
        tried += 1
        trial = {**gains, r["comp"]: r["scale"]}
        res = ev.run(trial, gen=True)
        st, g = axis_stats(ev.items, res, ev.axes), guard_kl(ev.items, res, base)
        gs = gen_similarity(ev.base_gen, res.meta["gen"])
        obj = objective(st, parts)
        if obj > cur and g <= budget and gs >= min_gen_sim:
            gains, cur = trial, obj
            best = {"axes": st, "guard_kl": g, "gen_sim": gs, "objective": obj}
            log(f"  keep {r['comp']}@{r['scale']:g}: " + " ".join(f"P({o})={st[a][o]['p']:.3f}" for a, o in parts)
                + f" guard KL {g:.3f} gen sim {gs:.2f}")
    if best is None:
        best = {"axes": axis_stats(ev.items, base, ev.axes), "guard_kl": 0.0, "gen_sim": 1.0, "objective": cur}
    best["wins"] = {f"{a}-{o}": max(best["axes"][a], key=lambda x: best["axes"][a][x]["p"]) == o for a, o in parts}
    return gains, best


def mix_gains(*gain_sets: dict[str, float], log=None) -> dict[str, float]:
    """Union of levers. A lever shared with the same gain is applied once (not squared);
    conflicting gains on one component multiply, with a warning."""
    out: dict[str, float] = {}
    for g in gain_sets:
        for c, s in g.items():
            if c in out and out[c] != s:
                if log:
                    log(f"  conflict on {c}: {out[c]:g} vs {s:g} -> {out[c] * s:g}")
                out[c] = out[c] * s
            else:
                out[c] = s
    return out


def evaluate(ev, gains: dict[str, float]) -> dict:
    res = ev.run(gains, gen=True)
    return {"axes": axis_stats(ev.items, res, ev.axes), "guard_kl": guard_kl(ev.items, res, ev.base_res),
            "gen_sim": gen_similarity(ev.base_gen, res.meta["gen"]), "gen": res.meta["gen"]}


def summary(score: dict) -> str:
    parts = []
    for a, opts in score["axes"].items():
        parts.append(f"{a}: " + " ".join(f"{o}={v['p']:.3f}" for o, v in opts.items()))
    g = f" | gen sim {score['gen_sim']:.2f}" if "gen_sim" in score else ""
    w = ""
    if "wins" in score:
        w = " | wins " + " ".join(f"{k.split('-', 1)[1]}{'✓' if v else '✗'}" for k, v in score["wins"].items())
    return " | ".join(parts) + f" | guard KL {score['guard_kl']:.3f}" + g + w


def all_mixes(axes: list[Axis]) -> list[tuple[str, ...]]:
    """Every combination of one non-AI value per axis."""
    per_axis = [[f"{a.name}-{o}" for o in a.targets()] for a in axes]
    return list(itertools.product(*per_axis))


# --- compatibility / apply --------------------------------------------------------------------

def check_compat(gguf: Path, overlay_path: Path, sha: str | None = None) -> dict:
    """exact: base sha256 matches. compatible: every edited tensor exists with the same type and
    shape (a different file with the same layout, e.g. a re-download or another quant run).
    incompatible: otherwise."""
    from wsp.gguf_io import GGUFModel

    ov = Overlay.load(overlay_path)
    if ov.base.sha256 and sha and sha == ov.base.sha256:
        return {"status": "exact", "detail": "sha256 matches the overlay's base"}
    m = GGUFModel(gguf)
    if ov.base.arch and m.arch != ov.base.arch:
        return {"status": "incompatible", "detail": f"arch {m.arch} != {ov.base.arch}"}
    ref = Path(REPO / "models" / ov.base.file)
    refm = GGUFModel(ref) if ref.exists() else None
    for e in ov.edits:
        t = m.tensors.get(e.tensor)
        if t is None:
            return {"status": "incompatible", "detail": f"missing tensor {e.tensor}"}
        if refm is not None:
            r = refm.tensors[e.tensor]
            if (t.qtype, t.shape) != (r.qtype, r.shape):
                return {"status": "incompatible", "detail": f"{e.tensor}: {t.qtype.name}{list(t.shape)} != {r.qtype.name}{list(r.shape)}"}
    note = "same tensor layout, different file: scores were measured on the base and may differ"
    if sha is None:
        note = "tensor layout matches (sha256 not checked)"
    return {"status": "compatible", "detail": note}


def apply_overlays(base: Path, overlay_paths: list[Path], out: Path) -> int:
    """Apply several overlays in order (scale edits on shared tensors multiply). Returns bytes written."""
    import shutil

    from wsp.gguf_io import GGUFModel
    from wsp.overlay.apply import compile_overlay, write_patch

    ovs = [Overlay.load(p) for p in overlay_paths]
    edits, seen = [], set()
    for ov in ovs:  # identical edits (a lever two overlays share) are applied once
        for e in ov.edits:
            key = e.model_dump_json(exclude={"note"})
            if key not in seen:
                seen.add(key)
                edits.append(e)
    merged = Overlay(base=ovs[0].base, meta={"parts": [p.stem for p in overlay_paths]}, edits=edits)
    model = GGUFModel(base)
    patch = compile_overlay(model, merged)
    shutil.copyfile(base, out)
    return write_patch(out, model, patch)
