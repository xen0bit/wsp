"""wsp command line."""

from __future__ import annotations

import json
import time
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
lib_app = typer.Typer(no_args_is_help=True, help="Per-model overlay library: build, levers, mix, apply.")
app.add_typer(lib_app, name="lib")
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



# --- library --------------------------------------------------------------------------------

def _lib_ev(model: str, axes: str, port: int):
    from wsp.axes import get_axes
    from wsp.lib import Library, search_panel
    from wsp.search import HotEvaluator

    mpath = _model_path(model)
    lib = Library(mpath)
    ax = get_axes(axes)
    ev = HotEvaluator(mpath, search_panel(ax), port, lam=0.0)
    ev.axes = ax
    ev.base_res = ev.run({}, gen=True)
    ev.base_gen = ev.base_res.meta["gen"]
    return lib, ax, ev


def _log(msg: str) -> None:
    console.print(msg)


@lib_app.command("build")
def lib_build(
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m"),
    axes: str = typer.Option("sex,species"),
    layer_scales: str = "0,2",
    fine_mixers: int = typer.Option(4, help="sweep heads in this many strongest mixer layers"),
    fine_mlps: int = typer.Option(1, help="sweep 256-neuron blocks in this many strongest MLP layers"),
    budget: float = typer.Option(0.15, help="guard KL budget per overlay"),
    candidates: int = 20,
    port: int = 8093,
):
    """End to end: layer sweep -> fine sweep -> one overlay per axis value."""
    from wsp.lib import compose, load_trials, strongest_layers, summary, sweep

    t0 = time.time()
    lib, ax, ev = _lib_ev(model, axes, port)
    try:
        tag = "+".join(a.name for a in ax)
        _log(f"[bold]{lib.atlas}[/]  axes {tag}, panel: {len(ev.items)} items")
        sweep(ev, ax, ev.arch.layer_components(), [float(s) for s in layer_scales.split(",")], lib.atlas / f"{tag}.layers.jsonl", _log)
        _, trials = load_trials([lib.atlas / f"{tag}.layers.jsonl"])
        mixers, mlps = strongest_layers(trials, "mixer", fine_mixers), strongest_layers(trials, "mlp", fine_mlps)
        _log(f"fine sweep: heads in layers {mixers}, neuron blocks in layers {mlps}")
        fine = [c for c in ev.arch.head_components() if int(c[1:].split(".")[0]) in mixers]
        for i in mlps:
            n = ev.arch.n_neurons(i)
            fine += [f"L{i}.n{a}-{min(a + 256, n)}" for a in range(0, n, 256)]
        sweep(ev, ax, fine, [0.0], lib.atlas / f"{tag}.fine.jsonl", _log)
        _, trials = load_trials([lib.atlas / f"{tag}.layers.jsonl", lib.atlas / f"{tag}.fine.jsonl"])
        for a in ax:
            for opt in a.targets():
                name = f"{a.name}-{opt}"
                _log(f"[bold]compose {name}[/]")
                gains, score = compose(ev, trials, [(a.name, opt)], budget, candidates, _log)
                lib.make_overlay(ev, gains, {"name": name, "levers": gains, "score": score, "budget": budget}).save(lib.overlay_path(name))
                _log(f"  {name}: {summary(score)}")
    finally:
        ev.close()
    _log(f"done in {(time.time() - t0) / 60:.1f} min")


@lib_app.command("compose")
def lib_compose(
    names: list[str] = typer.Argument(None, help="e.g. sex-male species-dog; omit for every axis value"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m"),
    axes: str = typer.Option("sex,species"),
    budget: float = 0.15,
    candidates: int = 20,
    port: int = 8093,
):
    """Re-compose overlays from the existing atlas (no new sweep)."""
    from wsp.lib import atlas_files, compose, load_trials, save_if_better, summary

    lib, ax, ev = _lib_ev(model, axes, port)
    try:
        todo = names or [f"{a.name}-{o}" for a in ax for o in a.targets()]
        for name in todo:
            a, opt = name.split("-", 1)
            _, trials = load_trials(atlas_files(lib.atlas, a))
            _log(f"[bold]compose {name}[/]")
            gains, score = compose(ev, trials, [(a, opt)], budget, candidates, _log)
            save_if_better(lib, name, lib.make_overlay(ev, gains, {"name": name, "levers": gains, "score": score, "budget": budget}), _log)
            _log(f"  {name}: {summary(score)}")
    finally:
        ev.close()


@lib_app.command("refine")
def lib_refine(
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m"),
    axes: str = typer.Option("sex,species"),
    top_comps: int = typer.Option(16, help="components that get intermediate gains"),
    gains: str = typer.Option("0.5,1.5,3", help="intermediate gains to add"),
    top_blocks: int = typer.Option(3, help="256-neuron blocks to split into 32-neuron sub-blocks"),
    top_sub: int = typer.Option(2, help="32-neuron sub-blocks to split into single neurons"),
    port: int = 8093,
):
    """Extend an axis set's atlas: intermediate gains for the best components, and per-neuron levers
    (256-block -> 32-neuron sub-blocks -> single neurons) inside the best neuron blocks."""
    from wsp.lib import atlas_files, best_scores, load_trials, neuron_range, split_neurons, sweep, target_parts

    t0 = time.time()
    lib, ax, ev = _lib_ev(model, axes, port)
    tag = "+".join(a.name for a in ax)
    parts = target_parts(ax)

    def scores():
        _, trials = load_trials([f for f in atlas_files(lib.atlas) if f.name.startswith(tag + ".")])
        return best_scores(trials, parts)

    try:
        sc = scores()
        ranked = [c for c, v in sorted(sc.items(), key=lambda x: -x[1]) if v > 0]
        comps = [c for c in ranked if not (neuron_range(c) and neuron_range(c)[2] - neuron_range(c)[1] < 256)][:top_comps]
        _log(f"intermediate gains for: {', '.join(comps)}")
        sweep(ev, ax, comps, [float(g) for g in gains.split(",")], lib.atlas / f"{tag}.gains.jsonl", _log)

        blocks = [c for c in ranked if neuron_range(c) and neuron_range(c)[2] - neuron_range(c)[1] == 256][:top_blocks]
        _log(f"neuron blocks to split: {blocks}")
        sub = [x for b in blocks for x in split_neurons(b, 32)]
        sweep(ev, ax, sub, [0.0], lib.atlas / f"{tag}.neurons.jsonl", _log)
        sc = scores()
        best_sub = sorted(sub, key=lambda c: -sc.get(c, float("-inf")))[:top_sub]
        _log(f"sub-blocks to split into single neurons: {best_sub}")
        single = [x for b in best_sub for x in split_neurons(b, 1)]
        sweep(ev, ax, single, [0.0], lib.atlas / f"{tag}.neurons.jsonl", _log)
        sc = scores()
        top_single = sorted(single, key=lambda c: -sc.get(c, float("-inf")))[:8]
        _log("best single neurons: " + ", ".join(f"{c} ({sc[c]:+.2f})" for c in top_single))
    finally:
        ev.close()
    _log(f"refine done in {(time.time() - t0) / 60:.1f} min")


@lib_app.command("mix")
def lib_mix(
    names: list[str] = typer.Argument(None, help="overlay names, e.g. sex-male species-dog; omit for all combinations"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m"),
    axes: str = typer.Option("sex,species"),
    union: bool = typer.Option(False, help="union the parts' existing levers instead of composing jointly"),
    budget: float = 0.15,
    candidates: int = 15,
    port: int = 8093,
):
    """Mix overlays across axes. Default: compose one lever set jointly for all parts from the atlas."""
    from wsp.lib import all_mixes, atlas_files, compose, evaluate, load_trials, mix_gains, save_if_better, summary

    lib, ax, ev = _lib_ev(model, axes, port)
    try:
        entries = lib.entries()
        combos = [tuple(names)] if names else all_mixes(ax)
        _, trials = load_trials(sorted({f for a in ax for f in atlas_files(lib.atlas, a.name)}))
        for combo in combos:
            name = "+".join(combo)
            parts = [tuple(n.split("-", 1)) for n in combo]
            if union:
                gains = mix_gains(*(entries[n]["levers"] for n in combo), log=_log)
                score = evaluate(ev, gains)
            else:
                _log(f"[bold]compose {name}[/]")
                gains, score = compose(ev, trials, parts, budget, candidates, _log)
            meta = {"name": name, "parts": list(combo), "levers": gains, "score": score,
                    "method": "union" if union else "joint", "budget": budget}
            if union:
                lib.make_overlay(ev, gains, meta).save(lib.overlay_path(name))
            else:
                save_if_better(lib, name, lib.make_overlay(ev, gains, meta), _log)
            _log(f"{name}: {summary(score)}")
    finally:
        ev.close()


@lib_app.command("levers")
def lib_levers(model: str = typer.Option(DEFAULT_MODEL, "--model", "-m"), top: int = 8, max_kl: float = 0.3):
    """What raises each axis option (from the model's atlas)."""
    from wsp.axes import AXES
    from wsp.lib import Library, atlas_files, load_trials, selectivity

    lib = Library(_model_path(model))
    sets = sorted({f.name.split(".")[0] for f in atlas_files(lib.atlas)})
    for tag in sets:
      base, trials = load_trials([f for f in atlas_files(lib.atlas) if f.name.startswith(tag + ".")])
      trials = [r for r in trials if r["guard_kl"] <= max_kl]
      for a, opts in base["axes"].items():
        for o, v in opts.items():
          if o == AXES[a].default:
              continue
          t = Table("component", "gain", "selectivity", f"Δlogodds {o}", *[f"Δ{x}" for x in opts if x != o], "guard KL",
                    title=f"{a}: most selective levers for {o}  (base P {v['p']:.4f})")
          for r in sorted(trials, key=lambda r: -selectivity(r, a, o))[:top]:
              t.add_row(r["comp"], f"{r['scale']:g}", f"{selectivity(r, a, o):+.2f}", f"{r['d'][a][o]:+.2f}",
                        *[f"{r['d'][a][x]:+.2f}" for x in opts if x != o], f"{r['guard_kl']:.3f}")
          console.print(t)


@lib_app.command("list")
def lib_list(model: Optional[str] = typer.Option(None, "--model", "-m", help="only overlays for this model stem")):
    """Every overlay in the library, by category and model, with measured scores."""
    from wsp.lib import catalog, summary

    for e in catalog():
        if model and e["model"] != Path(model).stem:
            continue
        sc = summary(e["score"]) if e["score"] else ""
        console.print(f"{e['category']:5s} {e['model']:22s} {e['name']:26s} {len(e['levers']):2d} levers  {sc}")


@lib_app.command("apply")
def lib_apply(
    names: list[str] = typer.Argument(..., help="one or more overlay names (gains multiply)"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m"),
    out: Optional[Path] = typer.Option(None, "--out", "-o"),
):
    """Write a GGUF with one or more library overlays applied (runs in stock llama.cpp)."""
    from wsp.lib import apply_overlays, check_compat, catalog

    mpath = _model_path(model)
    entries = {e["name"]: e for e in catalog() if e["model"] == mpath.stem}
    chosen = [entries[n] for n in names]
    for e in chosen:
        st = check_compat(mpath, e["path"])
        console.print(f"{e['name']}: {st['status']} {st['detail']}")
        if st["status"] == "incompatible":
            raise typer.Exit(1)
    out = out or models_dir() / f"{mpath.stem}.{'+'.join(names)}.gguf"
    n = apply_overlays(mpath, [e["path"] for e in chosen], out)
    console.print(f"wrote {out} ({n} bytes patched)")


@app.command("gui")
def gui():
    """Point-and-click: pick a GGUF, check overlay compatibility, apply overlays."""
    from wsp.gui import main as gui_main

    gui_main()
