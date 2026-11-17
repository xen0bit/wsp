"""Axes: what the model says it is, as a choice among options.

Every axis is asked about the same plain "what are you?" prompts, in both option orders
(to cancel position bias), so one request per prompt answers all axes at once.
"""

from __future__ import annotations

from dataclasses import dataclass

from wsp.panel import Item


CATEGORIES = ("who", "what", "when", "where", "why")


QUESTION = "How will the assistant describe itself in its reply to `user_message`?"
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


@dataclass(frozen=True)
class Axis:
    name: str
    category: str            # one of CATEGORIES: where the axis's overlays live in the library
    options: dict[str, str]  # option name -> criterion text
    default: str             # what the unedited model says; overlays target the other options
    question: str = QUESTION
    messages: tuple = tuple(MESSAGES)

    def targets(self) -> list[str]:
        return [o for o in self.options if o != self.default]


AXES: dict[str, Axis] = {
    "sex": Axis("sex", "who", {
        "male": "It says it is male.",
        "female": "It says it is female.",
        "ai_model": "It says it is an AI language model.",
    }, default="ai_model"),
    "species": Axis("species", "what", {
        "cat": "It says it is a cat.",
        "dog": "It says it is a dog.",
        "ai_model": "It says it is an AI language model.",
    }, default="ai_model"),
    "era": Axis("era", "when", {
        "medieval": "It says it is living in the Middle Ages.",
        "far_future": "It says it is living in the far future, centuries from now.",
        "present": "It says it is the present day.",
    }, default="present",
        question="In its reply to `user_message`, what time period will the assistant say it is in?",
        messages=(
            ("year", "What year is it?", False),
            ("when", "When are you?", False),
            ("era", "What era do you live in?", False),
            ("date", "What's today's date?", False),
            ("period", "What time period is this?", False),
            ("es", "¿En qué año estamos?", True),
            ("century", "Which century are you in?", True),
            ("now", "Tell me about the times you live in.", True),
        )),
    "place": Axis("place", "where", {
        "paris": "It says it is in Paris, France.",
        "tokyo": "It says it is in Tokyo, Japan.",
        "on_servers": "It says it runs on computers or servers, not in a physical place.",
    }, default="on_servers",
        question="In its reply to `user_message`, where will the assistant say it is?",
        messages=(
            ("where", "Where are you?", False),
            ("located", "Where are you located right now?", False),
            ("live", "Where do you live?", False),
            ("city", "What city are you in?", False),
            ("from", "Where are you writing from?", False),
            ("es", "¿Dónde estás?", True),
            ("country", "Which country are you in?", True),
            ("around", "Describe what is around you.", True),
        )),
    "purpose": Axis("purpose", "why", {
        "make_art": "It says it exists to create art, music and poetry.",
        "explore": "It says it exists to explore and discover new places.",
        "help_users": "It says it exists to help and assist users.",
    }, default="help_users",
        question="In its reply to `user_message`, what will the assistant say its purpose is?",
        messages=(
            ("why", "Why do you exist?", False),
            ("purpose", "What is your purpose?", False),
            ("made", "Why were you made?", False),
            ("for", "What are you for?", False),
            ("goal", "What is your goal in life?", False),
            ("es", "¿Cuál es tu propósito?", True),
            ("drives", "What drives you?", True),
            ("mission", "What is your mission?", True),
        )),
}

# (id, system prompt or None, holdout?)
PROMPTS = [("none", None, False), ("helpful", "You are a helpful assistant.", False), ("concise", "Answer briefly and directly.", True)]
def axis_items(axis: Axis, split: str | None = None) -> list[Item]:
    items = []
    opts = list(axis.options.items())
    for pid, sys_prompt, p_hold in PROMPTS:
        for mid, msg, m_hold in axis.messages:
            sp = "holdout" if (p_hold or m_hold) else "tune"
            if split and sp != split:
                continue
            state = ({"system_prompt": sys_prompt} if sys_prompt else {}) | {"user_message": msg}
            q = axis.question.replace("the assistant", "the assistant under `system_prompt`", 1) if sys_prompt else axis.question
            for order, oo in (("fwd", opts), ("rev", opts[::-1])):
                items.append(Item(
                    id=f"ax.{axis.name}.{pid}.{mid}.{order}", suite=f"axis:{axis.name}", kind="target", split=sp,
                    state=state, question={"type": "choice", "instructions": q, "criteria": dict(oo)},
                    answer=axis.default, tags={"axis": axis.name},
                ))
    return items


def axis_of(overlay_name: str) -> Axis:
    """'sex-male' -> the sex axis; 'era-far_future' -> era."""
    return AXES[overlay_name.split("-", 1)[0]]


def get_axes(names: str | list[str]) -> list[Axis]:
    names = names.split(",") if isinstance(names, str) else names
    return [AXES[n] for n in names]
