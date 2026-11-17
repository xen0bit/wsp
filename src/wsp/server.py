"""llama-server lifecycle and the /v1/systemone client.

Determinism: one slot (-np 1) and fixed batch sizes make repeated requests bit-identical,
including across restarts (checked on Qwen3.5-4B, CUDA build).
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from wsp.config import DEFAULT_PORT, llama_bin_dir, runs_dir


@dataclass
class ServerConfig:
    model: Path
    port: int = DEFAULT_PORT
    ctx: int = 8192
    n_gpu_layers: int = 99
    batch: int = 2048
    ubatch: int = 512
    permute: bool = False
    no_mmap: bool = False
    extra: list[str] = field(default_factory=list)

    def argv(self) -> list[str]:
        args = [
            str(llama_bin_dir() / "llama-server"),
            "-m", str(self.model),
            "--port", str(self.port),
            "-ngl", str(self.n_gpu_layers),
            "-np", "1",
            "-c", str(self.ctx),
            "-b", str(self.batch),
            "-ub", str(self.ubatch),
            "--no-webui",
            "-cram", "0",
        ]
        if self.permute:
            args.append("--systemone-permute")
        if self.no_mmap:
            args += ["--load-mode", "none"]
        return args + self.extra

    def fingerprint(self) -> dict:
        return {
            "model": self.model.name,
            "ctx": self.ctx,
            "batch": self.batch,
            "ubatch": self.ubatch,
            "permute": self.permute,
            "extra": self.extra,
        }


class LlamaServer:
    """Starts llama-server as a child process. Use as a context manager."""

    def __init__(self, cfg: ServerConfig, log: Path | None = None):
        self.cfg = cfg
        self.log = log or runs_dir() / f"server-{cfg.port}.log"
        self.proc: subprocess.Popen | None = None
        self.url = f"http://127.0.0.1:{cfg.port}"

    def start(self, timeout: float = 120.0) -> "LlamaServer":
        if _healthy(self.url):
            raise RuntimeError(f"something is already serving on {self.url}")
        logf = open(self.log, "w")
        self.proc = subprocess.Popen(self.cfg.argv(), stdout=logf, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with {self.proc.returncode}; see {self.log}")
            if _healthy(self.url):
                return self
            time.sleep(0.05)
        self.stop()
        raise TimeoutError(f"llama-server not healthy after {timeout}s; see {self.log}")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:  # known shutdown hang in the fork, see PR #1
                self.proc.kill()
                self.proc.wait()
        self.proc = None

    def restart(self, model: Path | None = None) -> "LlamaServer":
        self.stop()
        if model is not None:
            self.cfg.model = model
        return self.start()

    def __enter__(self) -> "LlamaServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def _healthy(url: str) -> bool:
    try:
        return httpx.get(f"{url}/health", timeout=1.0).status_code == 200
    except httpx.HTTPError:
        return False


class SystemOneError(RuntimeError):
    pass


class SystemOneClient:
    def __init__(self, url: str = f"http://127.0.0.1:{DEFAULT_PORT}", timeout: float = 300.0):
        self.url = url
        self.http = httpx.Client(timeout=timeout)

    def ask(self, state, questions: dict) -> dict:
        body = {"model": "jev-latest", "state": state, "questions": questions}
        r = self.http.post(f"{self.url}/v1/systemone", content=json.dumps(body))
        if r.status_code != 200:
            raise SystemOneError(f"{r.status_code}: {r.text[:500]}")
        return r.json()

    def close(self) -> None:
        self.http.close()
