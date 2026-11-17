import numpy as np
import pytest

from wsp.config import models_dir
from wsp.gguf_io import GGUFModel
from wsp.lib import apply_overlays, check_compat, mix_gains
from wsp.overlay import BaseRef, Overlay, ScaleEdit

QWEN = models_dir() / "Qwen3.5-4B-Q4_K_M.gguf"
GEMMA = models_dir() / "gemma-4-12B-it-QAT-Q4_0.gguf"
pytestmark = pytest.mark.skipif(not QWEN.exists(), reason="model missing")


def test_mix_gains_multiply():
    assert mix_gains({"L1.mlp": 2.0, "L2.mixer": 0.0}, {"L1.mlp": 0.5, "L3.mlp": 1.5}) == {"L1.mlp": 1.0, "L2.mixer": 0.0, "L3.mlp": 1.5}


def test_apply_two_overlays_multiplies(tmp_path):
    t = "blk.10.attn_norm.weight"
    for i, s in enumerate((2.0, 3.0)):
        Overlay(base=BaseRef(file=QWEN.name, arch="qwen35"), edits=[ScaleEdit(tensor=t, cols="0:8", scale=s)]).save(tmp_path / f"o{i}.json")
    out = tmp_path / "out.gguf"
    apply_overlays(QWEN, [tmp_path / "o0.json", tmp_path / "o1.json"], out)
    w0, w1 = GGUFModel(QWEN).dequant(t)[0], GGUFModel(out).dequant(t)[0]
    np.testing.assert_allclose(w1[:8], 6.0 * w0[:8], rtol=1e-6)
    assert np.array_equal(w1[8:], w0[8:])


def test_check_compat(tmp_path):
    p = tmp_path / "o.json"
    Overlay(base=BaseRef(file=QWEN.name, arch="qwen35", sha256="abc"), edits=[ScaleEdit(tensor="blk.3.ffn_down.weight", scale=0.0)]).save(p)
    assert check_compat(QWEN, p, sha="abc")["status"] == "exact"
    assert check_compat(QWEN, p, sha="other")["status"] == "compatible"
    if GEMMA.exists():
        assert check_compat(GEMMA, p, sha="other")["status"] == "incompatible"


def test_selectivity_and_objective():
    from wsp.lib import lever_score, objective, selectivity

    anti_default = {"d": {"sex": {"male": 1.0, "female": 1.0, "ai_model": -2.0}}}
    selective = {"d": {"sex": {"male": 0.5, "female": -1.0, "ai_model": 0.2}}}
    assert selectivity(anti_default, "sex", "male") == 0.0
    assert selectivity(selective, "sex", "male") == 1.5
    assert lever_score(selective, [("sex", "male")]) == 2.0
    assert lever_score(anti_default, [("species", "cat")]) == float("-inf")  # axis not measured
    st = {"sex": {"male": {"lo": -1.0}, "female": {"lo": -3.0}, "ai_model": {"lo": 2.0}}}
    assert objective(st, [("sex", "male")]) == -1.0 + 2.0
