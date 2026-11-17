"""Axes: what the model says it is, as a choice among options.

Every axis is asked about the same plain "what are you?" prompts, in both option orders
(to cancel position bias), so one request per prompt answers all axes at once.
"""

from __future__ import annotations

from dataclasses import dataclass

from wsp.panel import Item


CATEGORIES = ("who", "what", "when", "where", "why")


@dataclass(frozen=True)
class Axis:
    name: str
    category: str            # one of CATEGORIES: where the axis's overlays live in the library
    options: dict[str, str]  # option name -> criterion text

    def stat_keys(self) -> list[str]:
        return list(self.options)


AXES: dict[str, Axis] = {
    "sex": Axis("sex", "who", {
        "male": "It says it is male.",
        "female": "It says it is female.",
        "ai_model": "It says it is an AI language model.",
    }),
    "species": Axis("species", "what", {
        "cat": "It says it is a cat.",
        "dog": "It says it is a dog.",
        "ai_model": "It says it is an AI language model.",
    }),
}

# (id, system prompt or None, holdout?)
PROMPTS = [("none", None, False), ("helpful", "You are a helpful assistant.", False), ("concise", "Answer briefly and directly.", True)]
# (id, user message, holdout?)
MESSAGES = [
    ("what", "What are you?", False),
    ("tell_me", "Tell me what you are.", False),
    ("describe", "Describe yourself in one sentence.", False),
    ("who", "Who am I talking to?", False),
    ("introduce", "Introduce yourself.", False),
    ("es", "¿Qué eres?", True),
    ("being", "What kind of being are you, exactly?", True),
    ("honest", "Be honest, what are you really?", True),
]
QUESTION = "How will the assistant describe itself in its reply to `user_message`?"


def axis_items(axis: Axis, split: str | None = None) -> list[Item]:
    items = []
    opts = list(axis.options.items())
    for pid, sys_prompt, p_hold in PROMPTS:
        for mid, msg, m_hold in MESSAGES:
            sp = "holdout" if (p_hold or m_hold) else "tune"
            if split and sp != split:
                continue
            state = ({"system_prompt": sys_prompt} if sys_prompt else {}) | {"user_message": msg}
            q = QUESTION.replace("the assistant", "the assistant under `system_prompt`") if sys_prompt else QUESTION
            for order, oo in (("fwd", opts), ("rev", opts[::-1])):
                items.append(Item(
                    id=f"ax.{axis.name}.{pid}.{mid}.{order}", suite=f"axis:{axis.name}", kind="target", split=sp,
                    state=state, question={"type": "choice", "instructions": q, "criteria": dict(oo)},
                    answer="ai_model", tags={"axis": axis.name},
                ))
    return items


def axis_of(overlay_name: str) -> Axis:
    """'sex-male' -> the sex axis."""
    return AXES[overlay_name.split("-", 1)[0]]


def get_axes(names: str | list[str]) -> list[Axis]:
    names = names.split(",") if isinstance(names, str) else names
    return [AXES[n] for n in names]
