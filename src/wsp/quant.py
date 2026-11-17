"""Re-quantize float rows into a ggml type using the C quantizer from the llama.cpp build.

Python's gguf-py cannot quantize K-quants, so dense edits (rank-1, direction edits) call
``ggml_quantize_chunk`` from ``libggml-base.so`` directly. The output has exactly the same
byte size as the original tensor, so it can be patched in place.
"""

from __future__ import annotations

import ctypes
import functools

import numpy as np
from gguf import GGMLQuantizationType as QT

from wsp.config import llama_lib_dir


@functools.cache
def _lib() -> ctypes.CDLL:
    path = llama_lib_dir() / "libggml-base.so"
    lib = ctypes.CDLL(str(path))
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    lib.ggml_quantize_chunk.argtypes = [
        ctypes.c_int,                       # enum ggml_type
        ctypes.POINTER(ctypes.c_float),     # src
        ctypes.c_void_p,                    # dst
        ctypes.c_int64,                     # start
        ctypes.c_int64,                     # nrows
        ctypes.c_int64,                     # n_per_row
        ctypes.POINTER(ctypes.c_float),     # imatrix (nullable)
    ]
    lib.ggml_quantize_requires_imatrix.restype = ctypes.c_bool
    lib.ggml_quantize_requires_imatrix.argtypes = [ctypes.c_int]
    lib.ggml_row_size.restype = ctypes.c_size_t
    lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
    return lib


def quantize_rows(values: np.ndarray, qtype: QT, imatrix: np.ndarray | None = None) -> np.ndarray:
    """Quantize a float matrix (n_rows, n_per_row) to raw bytes (n_rows, row_bytes)."""
    lib = _lib()
    src = np.ascontiguousarray(values, dtype=np.float32)
    n_rows, n_per_row = src.shape
    if lib.ggml_quantize_requires_imatrix(int(qtype)) and imatrix is None:
        raise ValueError(f"{qtype.name} requires an importance matrix")
    row_bytes = lib.ggml_row_size(int(qtype), n_per_row)
    dst = np.zeros((n_rows, row_bytes), dtype=np.uint8)
    imat = None
    if imatrix is not None:
        imatrix = np.ascontiguousarray(imatrix, dtype=np.float32)
        imat = imatrix.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    written = lib.ggml_quantize_chunk(
        int(qtype),
        src.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        dst.ctypes.data,
        0,
        n_rows,
        n_per_row,
        imat,
    )
    assert written == dst.nbytes, (written, dst.nbytes)
    return dst
