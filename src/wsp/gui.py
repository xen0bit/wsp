"""wsp gui: pick a GGUF, see which library overlays fit it, apply one or more.

Run with ``uv run wsp gui``. Uses tkinter (bundled with Python), no extra dependencies.
"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from wsp.config import llama_bin_dir, models_dir

CHECK, EMPTY = "☑", "☐"
STATUS_TAG = {"exact": "exact", "compatible": "compatible", "incompatible": "incompatible", "…": "pending"}


def _score_text(score: dict | None) -> str:
    if not score:
        return ""
    parts = []
    for opts in score["axes"].values():
        top = max(opts.items(), key=lambda kv: kv[1]["p"])
        parts.append(" ".join(f"{o}={v['p']:.2f}" for o, v in opts.items() if o != "ai_model" or top[0] == o))
    return "  |  ".join(parts) + f"  |  KL {score['guard_kl']:.2f}"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("wsp — weight overlays")
        self.geometry("1100x620")
        self.model: Path | None = None
        self.sha: str | None = None
        self.entries: list[dict] = []
        self.checked: set[str] = set()
        self.status: dict[str, dict] = {}
        self._build()
        self._load_catalog()
        local = sorted(models_dir().glob("*.gguf"))
        if local:
            self.model_box["values"] = [p.name for p in local]

    # layout -------------------------------------------------------------------------------
    def _build(self):
        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")
        ttk.Label(top, text="Model:").pack(side="left")
        self.model_box = ttk.Combobox(top, width=60, state="readonly")
        self.model_box.pack(side="left", padx=4)
        self.model_box.bind("<<ComboboxSelected>>", lambda _e: self.select_model(models_dir() / self.model_box.get()))
        ttk.Button(top, text="Browse…", command=self.browse).pack(side="left", padx=4)
        self.model_info = ttk.Label(top, text="pick a GGUF to check which overlays fit", foreground="#555")
        self.model_info.pack(side="left", padx=10)

        cols = ("on", "category", "name", "model", "levers", "fit", "score")
        self.tree = ttk.Treeview(self, columns=cols, show="headings", selectmode="none")
        widths = {"on": 40, "category": 70, "name": 200, "model": 170, "levers": 55, "fit": 95, "score": 440}
        for c in cols:
            self.tree.heading(c, text="" if c == "on" else c)
            self.tree.column(c, width=widths[c], anchor="center" if c in ("on", "levers", "fit") else "w")
        self.tree.tag_configure("exact", foreground="#1b7f3a")
        self.tree.tag_configure("compatible", foreground="#a86b00")
        self.tree.tag_configure("incompatible", foreground="#999999")
        self.tree.tag_configure("pending", foreground="#555555")
        self.tree.bind("<Button-1>", self.toggle)
        self.tree.pack(fill="both", expand=True, padx=8)

        bottom = ttk.Frame(self, padding=8)
        bottom.pack(fill="x")
        self.apply_btn = ttk.Button(bottom, text="Apply selected → GGUF…", command=self.apply, state="disabled")
        self.apply_btn.pack(side="right")
        ttk.Button(bottom, text="Reload library", command=self._load_catalog).pack(side="right", padx=6)
        self.msg = ttk.Label(bottom, text="", foreground="#333")
        self.msg.pack(side="left", fill="x", expand=True)

    # catalog ------------------------------------------------------------------------------
    def _load_catalog(self):
        from wsp.lib import catalog

        self.entries = catalog()
        self.checked &= {self._key(e) for e in self.entries}
        self._refresh()
        if self.model:
            self._check_all()

    @staticmethod
    def _key(e: dict) -> str:
        return f"{e['category']}/{e['model']}/{e['name']}"

    def _refresh(self):
        self.tree.delete(*self.tree.get_children())
        for e in self.entries:
            k = self._key(e)
            st = self.status.get(k, {}).get("status", "…" if self.model else "")
            self.tree.insert("", "end", iid=k, tags=(STATUS_TAG.get(st, "pending"),), values=(
                CHECK if k in self.checked else EMPTY, e["category"], e["name"], e["model"], len(e["levers"]), st, _score_text(e["score"])))
        self.apply_btn["state"] = "normal" if self.model and self.checked else "disabled"

    def toggle(self, event):
        k = self.tree.identify_row(event.y)
        if not k:
            return
        st = self.status.get(k, {}).get("status")
        if st not in ("exact", "compatible"):
            self.msg["text"] = "pick a model first" if not self.model else f"{k}: {self.status.get(k, {}).get('detail', 'checking…')}"
            return
        self.checked ^= {k}
        self._refresh()
        self.msg["text"] = f"{len(self.checked)} selected: " + ", ".join(sorted(x.split('/')[-1] for x in self.checked))

    # model selection ------------------------------------------------------------------------
    def browse(self):
        p = filedialog.askopenfilename(initialdir=models_dir(), filetypes=[("GGUF", "*.gguf")])
        if p:
            self.select_model(Path(p))

    def select_model(self, path: Path):
        self.model, self.sha, self.status, self.checked = path, None, {}, set()
        self.model_box.set(path.name)
        self.model_info["text"] = "reading…"
        self._refresh()
        self._check_all()

    def _check_all(self):
        model = self.model

        def work():
            from wsp.gguf_io import GGUFModel, file_sha256
            from wsp.lib import check_compat

            try:
                m = GGUFModel(model)
                self.after(0, lambda: self.model_info.config(text=f"{m.arch}, {len(m.tensors)} tensors — hashing…"))
                sha = file_sha256(model)
                status = {self._key(e): check_compat(model, e["path"], sha) for e in self.entries}
            except Exception as ex:  # noqa: BLE001
                self.after(0, lambda: messagebox.showerror("wsp", f"cannot read {model.name}: {ex}"))
                return
            if model != self.model:
                return

            def done():
                self.sha, self.status = sha, status
                n_ok = sum(s["status"] != "incompatible" for s in status.values())
                self.model_info["text"] = f"{m.arch}, sha256 {sha[:12]}…  —  {n_ok}/{len(status)} overlays fit"
                self._refresh()
            self.after(0, done)

        threading.Thread(target=work, daemon=True).start()

    # apply ------------------------------------------------------------------------------------
    def apply(self):
        from wsp.lib import apply_overlays

        chosen = [e for e in self.entries if self._key(e) in self.checked]
        names = "+".join(e["name"] for e in chosen)
        out = filedialog.asksaveasfilename(initialdir=self.model.parent, initialfile=f"{self.model.stem}.{names}.gguf",
                                           defaultextension=".gguf", filetypes=[("GGUF", "*.gguf")])
        if not out:
            return
        self.apply_btn["state"] = "disabled"
        self.msg["text"] = f"writing {Path(out).name}…"
        model = self.model

        def work():
            try:
                n = apply_overlays(model, [e["path"] for e in chosen], Path(out))
                cmd = f"{llama_bin_dir() / 'llama-cli'} -m '{out}' -ngl 99 -cnv"
                self.after(0, lambda: (self.msg.config(text=f"wrote {Path(out).name} ({n / 1e6:.1f} MB patched)"),
                                       messagebox.showinfo("wsp", f"Wrote {out}\n\nRun it with:\n{cmd}")))
            except Exception as ex:  # noqa: BLE001
                self.after(0, lambda: messagebox.showerror("wsp", str(ex)))
            finally:
                self.after(0, self._refresh)

        threading.Thread(target=work, daemon=True).start()


def main() -> None:
    App().mainloop()
