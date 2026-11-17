import numpy as np
import pytest
from gguf.quants import dequantize

from wsp.config import models_dir
from wsp.gguf_io import GGUFModel, scale_blocks, scale_float
from wsp.overlay import parse_index

QWEN = models_dir() / "Qwen3.5-4B-Q4_K_M.gguf"
GEMMA = models_dir() / "gemma-4-12B-it-QAT-Q4_0.gguf"
needs = lambda p: pytest.mark.skipif(not p.exists(), reason=f"{p.name} not present")


def test_parse_index():
    assert parse_index(None, 5) is None
    assert parse_index("all", 5) is None
    assert parse_index("1:3,4", 5).tolist() == [1, 2, 4]
    assert parse_index([3, 1, 3], 5).tolist() == [1, 3]
    with pytest.raises(IndexError):
        parse_index("4:6", 5)


def test_scale_float_roundtrip():
    raw = np.arange(8, dtype=np.float32).reshape(1, 8).view(np.uint8)
    out = scale_float(raw, __import__("gguf").GGMLQuantizationType.F32, np.array([1, 2]), 2.0).view(np.float32)
    assert out.tolist() == [[0, 2, 4, 3, 4, 5, 6, 7]]


@pytest.mark.parametrize("path,name", [
    (QWEN, "blk.3.ffn_up.weight"),      # Q4_K
    (QWEN, "blk.3.ffn_down.weight"),    # Q6_K
    (QWEN, "blk.0.attn_qkv.weight"),    # Q5_K
    (GEMMA, "blk.3.attn_output.weight"),  # Q4_0
])
def test_block_scale_is_multiplicative(path, name):
    if not path.exists():
        pytest.skip("model missing")
    m = GGUFModel(path)
    t = m.tensors[name]
    raw = np.asarray(m.raw(name)[:4])
    w = dequantize(raw, t.qtype).reshape(4, -1)
    for s in (0.0, 0.5, -1.0, 1.7):
        w2 = dequantize(scale_blocks(raw, t.qtype, t.type_size, None, s), t.qtype).reshape(4, -1)
        np.testing.assert_allclose(w2, s * w, rtol=0, atol=2e-3 * np.abs(w).max() * max(abs(s), 1))
    # one block only
    w3 = dequantize(scale_blocks(raw, t.qtype, t.type_size, np.array([1]), 0.0), t.qtype).reshape(4, -1)
    bs = t.block_size
    assert np.all(w3[:, bs:2 * bs] == 0)
    assert np.array_equal(np.delete(w3, np.s_[bs:2 * bs], axis=1), np.delete(w, np.s_[bs:2 * bs], axis=1))
