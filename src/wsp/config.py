"""Paths and defaults. Override with environment variables."""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def llama_cpp_dir() -> Path:
    return Path(os.environ.get("WSP_LLAMA_CPP", Path.home() / "Projects" / "llama.cpp"))


def llama_bin_dir() -> Path:
    return Path(os.environ.get("WSP_LLAMA_BIN", llama_cpp_dir() / "build" / "bin"))


def llama_lib_dir() -> Path:
    return llama_bin_dir()


def models_dir() -> Path:
    return Path(os.environ.get("WSP_MODELS", REPO / "models"))


def runs_dir() -> Path:
    p = Path(os.environ.get("WSP_RUNS", REPO / "runs"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def panels_dir() -> Path:
    return REPO / "panels"


DEFAULT_MODEL = "Qwen3.5-4B-Q4_K_M.gguf"
DEFAULT_PORT = 8091
