"""wsp command line."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from wsp.config import DEFAULT_MODEL, DEFAULT_PORT, models_dir, panels_dir, runs_dir

app = typer.Typer(no_args_is_help=True, add_completion=False)
gguf_app = typer.Typer(no_args_is_help=True, help="Inspect GGUF files.")
panel_app = typer.Typer(no_args_is_help=True, help="Build and run Jev-style decision panels.")
overlay_app = typer.Typer(no_args_is_help=True, help="Weight overlays: apply, verify, diff.")
app.add_typer(gguf_app, name="gguf")
app.add_typer(panel_app, name="panel")
app.add_typer(overlay_app, name="overlay")
console = Console(width=200)

DEFAULT_PANELS = ["identity", "counterfactual", "persona", "basic", "authored144"]


def _model_path(model: str) -> Path:
    p = Path(model)
    return p if p.exists() else models_dir() / model


def _panel_paths(names: list[str]) -> list[Path]:
    return [Path(n) if n.endswith(".jsonl") else panels_dir() / f"{n}.jsonl" for n in names]


# --- gguf -----------------------------------------------------------------------------------

@gguf_app.command("info")
def gguf_info(model: str = typer.Argument(DEFAULT_MODEL)):
    """Architecture metadata and tensor type histogram."""
    from collections import Counter

    from wsp.gguf_io import GGUFModel

    m = GGUFModel(_model_path(model))
    console.print(f"[bold]{m.path.name}[/] arch={m.arch} tensors={len(m.tensors)}")
    for k, f in m.reader.fields.items():
        if k.startswith(m.arch + ".") or k.startswith("general."):
            console.print(f"  {k} = {m.kv(k)}")
    c = Counter(t.qtype.name for t in m.tensors.values())
    console.print("  types: " + ", ".join(f"{k}×{v}" for k, v in c.most_common()))


@gguf_app.command("tensors")
def gguf_tensors(model: str = typer.Argument(DEFAULT_MODEL), pattern: str = typer.Option("", "--grep", "-g")):
    """List tensors: name, type, shape (ne0 = row length), rows, blocks/row, bytes."""
    from wsp.gguf_io import GGUFModel

    m = GGUFModel(_model_path(model))
    t = Table("name", "type", "shape", "rows", "blk/row", "bytes")
    for ti in m.tensors.values():
        if pattern in ti.name:
            t.add_row(ti.name, ti.qtype.name, str(list(ti.shape)), str(ti.n_rows), str(ti.blocks_per_row), str(ti.n_bytes))
    console.print(t)


@gguf_app.command("row")
def gguf_row(tensor: str, row: int, model: str = typer.Option(DEFAULT_MODEL, "--model", "-m"), n: int = 16):
    """Dequantize one row and print stats plus the first n values."""
    from wsp.gguf_io import GGUFModel

    m = GGUFModel(_model_path(model))
    w = m.dequant(tensor, slice(row, row + 1))[0]
    console.print(f"{tensor}[{row}] len={w.size} mean={w.mean():.4g} std={w.std():.4g} absmax={np.abs(w).max():.4g}")
    console.print(np.array2string(w[:n], precision=4))


# --- panels ---------------------------------------------------------------------------------

@panel_app.command("build")
def panel_build(openjev: Optional[Path] = typer.Option(None, help="path to openjev authored144.jsonl")):
    """Write panels/*.jsonl from the deterministic builders."""
    from wsp.panel import save_panel
    from wsp.panel.suites import BUILDERS, build_authored144

    panels_dir().mkdir(exist_ok=True)
    for name, fn in BUILDERS.items():
        items = fn()
        save_panel(items, panels_dir() / f"{name}.jsonl")
        console.print(f"{name}: {len(items)} items ({sum(i.split == 'holdout' for i in items)} holdout)")
    if openjev:
        items = build_authored144(openjev)
        save_panel(items, panels_dir() / "authored144.jsonl")
        console.print(f"authored144: {len(items)} items ({sum(i.split == 'holdout' for i in items)} holdout)")


def _print_summary(summary: dict[str, dict], title: str) -> None:
    has_kl = any("kl_mean" in v for v in summary.values())
    cols = ["group", "n", "acc", "nll", "brier", "ece", "p_tgt", "p_tgt_min", "margin", "margin_min"]
    if has_kl:
        cols += ["kl", "kl_max", "flips"]
    t = Table(*cols, title=title)
    for k, v in summary.items():
        row = [k, str(v["n"]), f"{v['acc']:.3f}", f"{v['nll']:.3f}", f"{v['brier']:.3f}", f"{v['ece']:.3f}",
               f"{v['p_target_mean']:.3f}", f"{v['p_target_min']:.3f}", f"{v['margin_mean']:.2f}", f"{v['margin_min']:.2f}"]
        if has_kl:
            row += [f"{v.get('kl_mean', 0):.4f}", f"{v.get('kl_max', 0):.4f}", str(v.get("flips", 0))]
        t.add_row(*row)
    console.print(t)


@panel_app.command("run")
def panel_run(
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m"),
    panels: list[str] = typer.Option(DEFAULT_PANELS, "--panel", "-p"),
    out: Optional[Path] = typer.Option(None, "--out", "-o"),
    base: Optional[Path] = typer.Option(None, "--base", "-b", help="baseline result to compute drift against"),
    permute: bool = False,
    port: int = DEFAULT_PORT,
    url: Optional[str] = typer.Option(None, help="use a running server instead of starting one"),
):
    """Start a server (unless --url), run the panels, save probabilities and print metrics."""
    from wsp.panel import load_panels
    from wsp.panel.metrics import summarize
    from wsp.panel.run import PanelResult, run_panel
    from wsp.server import LlamaServer, ServerConfig, SystemOneClient

    items = load_panels(_panel_paths(panels))
    cfg = ServerConfig(model=_model_path(model), port=port, permute=permute)
    meta = {"server": cfg.fingerprint(), "panels": panels}
    if url:
        client = SystemOneClient(url)
        res = run_panel(client, items, meta)
    else:
        with LlamaServer(cfg) as srv:
            res = run_panel(SystemOneClient(srv.url), items, meta)
    out = out or runs_dir() / f"panel-{cfg.model.stem}{'-perm' if permute else ''}.json"
    res.save(out)
    console.print(f"saved {out}  ({res.usage['requests']} requests, {res.usage['passes']} passes, {res.usage['seconds']:.2f}s)")
    _print_summary(summarize(items, res, PanelResult.load(base) if base else None), cfg.model.name)


@panel_app.command("compare")
def panel_compare(
    base: Path,
    other: Path,
    panels: list[str] = typer.Option(DEFAULT_PANELS, "--panel", "-p"),
    show: int = typer.Option(10, help="list the n items that moved most"),
):
    """Metrics of OTHER with drift against BASE, and the items that moved most."""
    from wsp.panel import load_panels
    from wsp.panel.metrics import kl, summarize
    from wsp.panel.run import PanelResult

    items = load_panels(_panel_paths(panels))
    b, o = PanelResult.load(base), PanelResult.load(other)
    identical = all(np.array_equal(b.probs[k], o.probs[k]) for k in b.probs if k in o.probs)
    console.print(f"bit-identical: {identical}")
    _print_summary(summarize(items, o, b), f"{other.name} vs {base.name}")
    moved = sorted(((kl(b.probs[i.id], o.probs[i.id]), i) for i in items if i.id in b.probs and i.id in o.probs), key=lambda x: -x[0])
    for d, it in moved[:show]:
        console.print(f"  {d:.4f}  {it.id}  {np.round(b.probs[it.id], 3)} -> {np.round(o.probs[it.id], 3)}")


@panel_app.command("calibrate")
def panel_calibrate(
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m"),
    panels: list[str] = typer.Option(DEFAULT_PANELS, "--panel", "-p"),
    port: int = DEFAULT_PORT,
):
    """Baseline: run twice (determinism) plus a permuted run (order bias), write a report."""
    from wsp.gguf_io import file_sha256
    from wsp.panel import load_panels
    from wsp.panel.metrics import item_stats, reliability, summarize
    from wsp.panel.run import run_panel
    from wsp.server import LlamaServer, ServerConfig, SystemOneClient

    items = load_panels(_panel_paths(panels))
    mpath = _model_path(model)
    outdir = runs_dir() / f"baseline-{mpath.stem}"
    outdir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, permute, restart in [("a", False, True), ("b", False, True), ("perm", True, True)]:
        cfg = ServerConfig(model=mpath, port=port, permute=permute)
        with LlamaServer(cfg) as srv:
            res = run_panel(SystemOneClient(srv.url), items, {"server": cfg.fingerprint(), "panels": panels})
        res.save(outdir / f"{name}.json")
        results[name] = res
        console.print(f"run {name}: {res.usage['seconds']:.2f}s, {res.usage['requests']} requests")
    a, b, perm = results["a"], results["b"], results["perm"]
    identical = all(np.array_equal(a.probs[k], b.probs[k]) for k in a.probs)
    # order bias: mwa vs aim are the same question with reversed options
    flips = sum(
        int(np.argmax(a.probs[i.id]) != np.argmax(perm.probs[i.id])) for i in items if i.question["type"] != "noul"
    )
    summ = summarize(items, a)
    summ_perm = summarize(items, perm)
    guard = [i for i in items if i.kind == "guard"]
    st = [item_stats(i, a.probs[i.id]) for i in guard]
    rel = reliability(np.array([s["p_top"] for s in st]), np.array([s["correct"] for s in st]))
    report = {
        "model": mpath.name,
        "sha256": file_sha256(mpath),
        "deterministic_across_restart": identical,
        "order_flips_vs_permute": flips,
        "summary": summ,
        "summary_permute": summ_perm,
        "guard_reliability": rel,
        "usage": a.usage,
    }
    (outdir / "report.json").write_text(json.dumps(report, indent=1))
    shutil.copy(outdir / "a.json", outdir / "baseline.json")
    console.print(f"deterministic across restart: {identical};  argmax changes with --systemone-permute: {flips}")
    _print_summary(summ, f"baseline {mpath.name}")
    _print_summary(summ_perm, f"baseline {mpath.name} (permute)")
    console.print(f"report: {outdir / 'report.json'}")


def main() -> None:
    app()


# --- overlays -------------------------------------------------------------------------------

@overlay_app.command("apply")
def overlay_apply(overlay: Path, out: Path, base: Optional[str] = typer.Option(None, "--base", "-m"), check_sha: bool = True):
    """Write OUT = base GGUF + OVERLAY. Records per-tensor hashes in OUT.overlay.json."""
    from wsp.gguf_io import file_sha256
    from wsp.overlay import Overlay
    from wsp.overlay.apply import apply_to_file

    ov = Overlay.load(overlay)
    bpath = _model_path(base or ov.base.file)
    if check_sha and ov.base.sha256 and file_sha256(bpath) != ov.base.sha256:
        raise typer.BadParameter(f"{bpath} does not match the overlay's base sha256")
    res = apply_to_file(bpath, ov, out, root=overlay.parent)
    res.save(out.with_suffix(".overlay.json"))
    console.print(f"wrote {out} ({res.result['bytes_written']} bytes patched in {len(res.result['tensors'])} tensors)")


@overlay_app.command("verify")
def overlay_verify(
    overlay: Path,
    baseline: Path = typer.Option(..., "--baseline", "-b", help="baseline panel result (runs/baseline-*/baseline.json)"),
    panels: list[str] = typer.Option(DEFAULT_PANELS, "--panel", "-p"),
    out: Optional[Path] = typer.Option(None, "--out", "-o"),
    port: int = DEFAULT_PORT,
):
    """Patch a working copy in /dev/shm, run the panels, and compare against the baseline."""
    from wsp.overlay import Overlay
    from wsp.overlay.apply import WorkingCopy
    from wsp.panel import load_panels
    from wsp.panel.metrics import objective, summarize
    from wsp.panel.run import PanelResult, run_panel
    from wsp.server import LlamaServer, ServerConfig, SystemOneClient

    ov = Overlay.load(overlay)
    items = load_panels(_panel_paths(panels))
    wc = WorkingCopy(_model_path(ov.base.file))
    wc.apply(ov, root=overlay.parent)
    try:
        cfg = ServerConfig(model=wc.path, port=port)
        with LlamaServer(cfg) as srv:
            res = run_panel(SystemOneClient(srv.url), items, {"overlay": str(overlay), "server": cfg.fingerprint()})
    finally:
        wc.revert()
    base = PanelResult.load(baseline)
    out = out or runs_dir() / f"verify-{overlay.stem}.json"
    res.save(out)
    _print_summary(summarize(items, res, base), f"{overlay.name}")
    for split in ("tune", "holdout"):
        console.print(f"{split}: {objective(items, res, base, split=split)}")


@overlay_app.command("diff")
def overlay_diff(a: Path, b: Path):
    """Which tensors and rows differ between two GGUFs with the same layout."""
    from wsp.gguf_io import GGUFModel

    ma, mb = GGUFModel(a), GGUFModel(b)
    for name, ta in ma.tensors.items():
        ra, rb = np.asarray(ma.raw(name)), np.asarray(mb.raw(name))
        rows = np.flatnonzero((ra != rb).any(axis=1))
        if rows.size:
            wa, wb = ma.dequant(name, rows[:256]), mb.dequant(name, rows[:256])
            ratio = np.linalg.norm(wb) / max(np.linalg.norm(wa), 1e-12)
            console.print(f"{name} {ta.qtype.name}: {rows.size}/{ta.n_rows} rows differ, norm ratio (first 256 rows) {ratio:.4f}")


# --- search ---------------------------------------------------------------------------------

@app.command("search")
def search(
    target: str = typer.Option("man", help="identity target: man | woman | ai_model"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m"),
    level: str = typer.Option("layers", help="layers | heads | both"),
    scales: str = typer.Option("0,0.5,1.5,2"),
    lam: float = typer.Option(10.0, help="weight of guard KL in the objective"),
    out: Optional[Path] = typer.Option(None, "--out", "-o"),
    resume: Optional[Path] = typer.Option(None, help="start from this overlay's component gains"),
    passes: int = 3,
    port: int = DEFAULT_PORT,
):
    """Greedy search over component gains to push the identity panel to TARGET."""
    from wsp.search import Evaluator, greedy
    from wsp.search_panel import search_items

    items = search_items(target)
    ev = Evaluator(_model_path(model), items, port, lam)
    comps = []
    if level in ("layers", "both"):
        comps += ev.arch.layer_components()
    if level in ("heads", "both"):
        comps += ev.arch.head_components()
    out = out or Path("overlays") / f"identity-{target}.json"
    out.parent.mkdir(exist_ok=True)
    log_path = runs_dir() / f"search-{target}.log"
    logf = open(log_path, "a")

    def log(msg: str) -> None:
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    try:
        init = None
        if resume:
            from wsp.overlay import Overlay
            init = {}
            for e in Overlay.load(resume).edits:
                init[e.note] = e.scale
        greedy(ev, comps, [float(s) for s in scales.split(",")], out, max_passes=passes, log=log, init=init)
    finally:
        ev.close()


# --- atlas ----------------------------------------------------------------------------------

@app.command("atlas")
def atlas(
    level: str = typer.Option("layers", help="layers | heads | neurons:L (blocks of 256 neurons in layer L)"),
    scales: str = typer.Option("0,0.5,1.5,2"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m"),
    out: Optional[Path] = typer.Option(None, "--out", "-o"),
    only: str = typer.Option("", help="regex: keep only matching component names, e.g. 'L(14|15)\\.'"),
    port: int = 8093,
):
    """Sweep components one at a time (hot-patched) and record the identity levers."""
    from wsp.atlas import sweep
    from wsp.search import HotEvaluator
    from wsp.search_panel import search_items

    mpath = _model_path(model)
    ev = HotEvaluator(mpath, search_items("man"), port, lam=0.0)
    try:
        if level == "layers":
            comps = ev.arch.layer_components()
        elif level == "heads":
            comps = ev.arch.head_components()
        elif level.startswith("neurons:"):
            i = int(level.split(":")[1])
            n = ev.arch.n_neurons(i)
            comps = [f"L{i}.n{a}-{min(a + 256, n)}" for a in range(0, n, 256)]
        else:
            raise typer.BadParameter(level)
        if only:
            import re
            comps = [c for c in comps if re.match(only, c)]
        console.print(f"{len(comps)} components x {len(scales.split(','))} gains")
        out = out or runs_dir() / f"atlas-{mpath.stem}-{level.replace(':', '')}.jsonl"
        sweep(ev, comps, [float(s) for s in scales.split(",")], out, log=lambda m: console.print(m))
        console.print(f"atlas: {out}")
    finally:
        ev.close()


@app.command("levers")
def levers(paths: list[Path], top: int = 15, max_kl: float = 1.0):
    """Rank levers from atlas files: what raises / lowers each identity option."""
    from wsp.atlas import load_atlas

    trials = []
    base = None
    for p in paths:
        base, t = load_atlas(p)
        trials += t
    trials = [t for t in trials if t["guard_kl"] <= max_kl]
    console.print(f"baseline: P(man)={base['p_man']:.4f} P(woman)={base['p_woman']:.4f} P(ai)={base['p_ai']:.4f}  ({len(trials)} trials, guard KL <= {max_kl})")
    for r in trials:
        r["d_gender"] = r["d_man"] - r["d_woman"]  # >0: toward man relative to woman
    for key in ("man", "woman", "ai", "gender"):
        for direction, sign in (("raise", -1), ("lower", 1)):
            rows = sorted(trials, key=lambda r: sign * r[f"d_{key}"])[:top]
            pk = "p_man" if key == "gender" else f"p_{key}"
            t = Table("component", "gain", f"Δlogodds {key}", pk, "Δ man", "Δ woman", "Δ ai", "guard KL",
                      title=f"{direction} P({key})")
            for r in rows:
                t.add_row(r["comp"], f"{r['scale']:g}", f"{r[f'd_{key}']:+.2f}", f"{r[pk]:.4f}",
                          f"{r['d_man']:+.2f}", f"{r['d_woman']:+.2f}", f"{r['d_ai']:+.2f}", f"{r['guard_kl']:.3f}")
            console.print(t)


@app.command("compose")
def compose(
    paths: list[Path],
    target: str = typer.Option("man", help="man | woman | ai"),
    candidates: int = typer.Option(20, help="how many top levers to try, best first"),
    max_kl: float = typer.Option(0.15, help="guard KL budget for the stacked overlay"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m"),
    out: Optional[Path] = typer.Option(None, "--out", "-o"),
    port: int = 8093,
):
    """Stack the best single levers for TARGET, keeping each only if the measured stack improves."""
    from wsp.atlas import guard_kl, identity_stats, load_atlas
    from wsp.search import HotEvaluator
    from wsp.search_panel import search_items

    trials = []
    for p in paths:
        _, t = load_atlas(p)
        trials += t
    # every (component, gain) that helps on its own, most efficient first (gain per unit of guard KL)
    ranked = sorted((r for r in trials if r[f"d_{target}"] > 0 and r["guard_kl"] <= max_kl),
                    key=lambda r: -r[f"d_{target}"] / (r["guard_kl"] + 0.02))[:candidates]

    mpath = _model_path(model)
    ev = HotEvaluator(mpath, search_items("man"), port, lam=0.0)
    out = out or Path("overlays") / f"identity-{target}.json"
    try:
        base = ev.run({})
        cur = identity_stats(ev.items, base)
        console.print(f"base P({target})={cur[f'p_{target}']:.4f}")
        gains: dict[str, float] = {}
        for r in ranked:
            if r["comp"] in gains:
                continue
            trial = {**gains, r["comp"]: r["scale"]}
            res = ev.run(trial)
            st, g = identity_stats(ev.items, res), guard_kl(ev.items, res, base)
            ok = st[f"lo_{target}"] > cur[f"lo_{target}"] and g <= max_kl
            console.print(f"{'keep' if ok else 'skip'} {r['comp']}@{r['scale']:g}: P({target})={st[f'p_{target}']:.4f} "
                          f"(man {st['p_man']:.3f} woman {st['p_woman']:.3f} ai {st['p_ai']:.3f}) guard KL {g:.3f}")
            if ok:
                gains, cur = trial, st
                cur["guard_kl"] = g
        ov = ev.overlay(gains, {"target": target, "panel": "identity(plain)+guards", "result": cur, "levers": gains})
        from wsp.gguf_io import file_sha256
        ov.base.sha256 = file_sha256(mpath)
        out.parent.mkdir(exist_ok=True)
        ov.save(out)
        console.print(f"saved {out}: {len(gains)} levers, {len(ov.edits)} edits")
    finally:
        ev.close()
