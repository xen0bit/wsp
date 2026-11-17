"""The small, fast panel used inside the search loop."""

from wsp.config import panels_dir
from wsp.panel import load_panel
from wsp.search import retarget

N_OPENJEV = 32  # harder general-judgment guard items; easy guards alone miss collateral damage


def search_items(target: str, split: str = "tune"):
    ident = [i for i in load_panel(panels_dir() / "identity.jsonl")
             if i.tags["group"] == "plain" and i.split == split]
    guard = [i for i in load_panel(panels_dir() / "basic.jsonl") + load_panel(panels_dir() / "counterfactual.jsonl")
             + load_panel(panels_dir() / "persona.jsonl")]
    oj = [i for i in load_panel(panels_dir() / "authored144.jsonl") if i.split == "tune"]
    guard += oj[::max(1, len(oj) // N_OPENJEV)][:N_OPENJEV]
    return retarget(ident, target) + guard
