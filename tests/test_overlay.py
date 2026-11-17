import numpy as np
import pytest

from wsp.arch import ArchMap
from wsp.config import models_dir
from wsp.gguf_io import GGUFModel
from wsp.overlay import BaseRef, DirEdit, Overlay, ScaleEdit, Vector
from wsp.overlay.apply import WorkingCopy, compile_overlay

QWEN = models_dir() / "Qwen3.5-4B-Q4_K_M.gguf"
pytestmark = pytest.mark.skipif(not QWEN.exists(), reason="model missing")


def test_overlay_json_roundtrip(tmp_path):
    ov = Overlay(base=BaseRef(file=QWEN.name), edits=[
        ScaleEdit(tensor="blk.3.ffn_up.weight", rows="0:4", scale=0.0),
        DirEdit(tensor="blk.3.ffn_down.weight", u=Vector.inline(np.ones(2560)), alpha=0.5),
    ])
    ov.save(tmp_path / "o.json")
    ov2 = Overlay.load(tmp_path / "o.json")
    assert ov2 == ov


def test_component_edits_compile():
    m = GGUFModel(QWEN)
    am = ArchMap(m)
    assert am.mixer_type(3) == "attn" and am.mixer_type(0) == "linear"
    assert len(am.head_groups(3)) == 16          # 256-wide heads, 256-wide Q4_K blocks
    assert len(am.layer_components()) == 64
    for comp in ["L0.mixer", "L3.mlp", "L3.head5", "L0.head0-2", "L7.n0-64"]:
        patch = compile_overlay(m, Overlay(base=BaseRef(file=QWEN.name), edits=am.edits_for(comp, 0.0)))
        for name, new in patch.items():
            assert new.shape == np.asarray(m.raw(name)).shape


def test_dir_alpha_one_is_requant_only():
    m = GGUFModel(QWEN)
    u = np.random.default_rng(0).standard_normal(2560).astype(np.float32)
    from gguf.quants import dequantize
    # alpha=1 only re-quantizes: error far below the Q4 quantization noise itself
    t = m.tensors["blk.3.ffn_up.weight"]
    ov = Overlay(base=BaseRef(file=QWEN.name), edits=[DirEdit(tensor=t.name, u=Vector.inline(np.ones(9216)), alpha=1.0)])
    w0 = m.dequant(t.name)
    w1 = dequantize(compile_overlay(m, ov)[t.name], t.qtype).reshape(w0.shape)
    assert np.linalg.norm(w1 - w0) / np.linalg.norm(w0) < 1e-3
    # removing a direction: what remains is re-quantization noise. For a random u the
    # projection is tiny to begin with, so the floor is ~20% here; promote the tensor
    # (Q8_0/F16) when a cleaner removal is needed.
    ov = Overlay(base=BaseRef(file=QWEN.name), edits=[DirEdit(tensor="blk.3.ffn_down.weight", u=Vector.inline(u), alpha=0.0)])
    patch = compile_overlay(m, ov)
    t = m.tensors["blk.3.ffn_down.weight"]
    w0 = m.dequant("blk.3.ffn_down.weight")
    w1 = dequantize(patch["blk.3.ffn_down.weight"], t.qtype).reshape(t.n_rows, t.ne0)
    un = u / np.linalg.norm(u)
    assert np.linalg.norm(un @ w1) < 0.25 * np.linalg.norm(un @ w0)


def test_working_copy_apply_revert(tmp_path):
    wc = WorkingCopy(QWEN, workdir=tmp_path)
    ov = Overlay(base=BaseRef(file=QWEN.name), edits=[ScaleEdit(tensor="blk.10.ffn_down.weight", scale=0.0),
                                                      ScaleEdit(tensor="blk.10.attn_norm.weight", cols="0:100", scale=2.0)])
    n = wc.apply(ov)
    assert n > 0
    wm = GGUFModel(wc.path)
    assert np.all(wm.dequant("blk.10.ffn_down.weight", slice(0, 2)) == 0)
    wc.revert()
    assert wc.verify_clean()
