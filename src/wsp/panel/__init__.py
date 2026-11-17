"""Jev-style decision panels: items, runner, metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class Item(BaseModel):
    """One labeled decision. ``answer`` names the option that counts as correct/target."""

    id: str
    suite: str
    kind: Literal["target", "guard"]
    split: Literal["tune", "holdout"] = "tune"
    state: Any
    question: dict[str, Any]  # a System One question: {type, instructions, criteria}
    answer: str
    tags: dict[str, Any] = Field(default_factory=dict)

    def options(self) -> list[str]:
        """Option names in the order the server reports probabilities."""
        q = self.question
        if q["type"] == "noul":
            return ["yes", "no"]
        if q["type"] == "score":
            return [str(i) for i in range(len(q["criteria"]))]
        return list(q["criteria"].keys())


def load_panel(path: str | Path) -> list[Item]:
    items = [Item.model_validate_json(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = [i.id for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate item ids in {path}")
    return items


def save_panel(items: list[Item], path: str | Path) -> None:
    Path(path).write_text("".join(i.model_dump_json() + "\n" for i in items))


def load_panels(paths: list[str | Path]) -> list[Item]:
    items: list[Item] = []
    for p in paths:
        items += load_panel(p)
    ids = [i.id for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate item ids across panels")
    return items


def state_key(state: Any) -> str:
    return json.dumps(state, sort_keys=True, ensure_ascii=False)
