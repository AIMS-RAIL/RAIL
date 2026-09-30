"""Model-facing prompt wrappers used in the paper's evaluation.

Every item already carries its task prompt (`item["prompt"]`). In the paper, models were asked
to answer in a `Reason: ...; Answer: ...` line so the answer can be parsed and the reasoning
length measured (needed for B-AUC). Two wrappers were used:

* Reasoning (Gf) items: open reasoning, no length cap
  (the paper's reasoning runner).
* All other capabilities: reasoning capped at N words (20 in the paper) and the allowed labels
  listed when the label set is small
  (the paper's efficiency and memory runner).
"""

from __future__ import annotations

from typing import List

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LISTEN_PREFIX = "Listen to the audio and solve the task"


def allowed_labels(item: dict) -> List[str]:
    if item["answer_format"] == "multiple_choice":
        return list(LETTERS[: len(item["choices"])])
    if item["answer_format"] == "closed_label":
        return list(item["choices"])
    return []


def _reasoning_wrapper(prompt: str) -> str:
    for prefix in (LISTEN_PREFIX + ": ", LISTEN_PREFIX + ".\n", LISTEN_PREFIX + "."):
        if prompt.startswith(prefix):
            prompt = prompt[len(prefix):].strip()
            break
    return (
        "Listen to the audio and solve the task.\n"
        + prompt
        + "\n\nReturn exactly one line in this format:\n"
        + "Reason: <your reasoning>; "
        + "Answer: <final answer>\n"
        + "Do not output anything else."
    )


def _capped_wrapper(prompt: str, allowed: List[str], reason_max_words: int) -> str:
    words = max(1, int(reason_max_words))
    include_allowed = bool(allowed) and len(allowed) <= 16 and sum(len(x) for x in allowed) <= 600
    head = (
        "Listen to the audio and solve the task.\n"
        + prompt
        + "\n\n"
        + "Return exactly one line in this format:\n"
        + f"Reason: <short reasoning, <= {words} words>; "
    )
    if include_allowed:
        return (
            head
            + "Answer: <allowedlabel only>\n"
            + "Do not output anything else.\n\n"
            + "Allowed labels:\n"
            + ", ".join(allowed)
        )
    return head + "Answer: <final answer>\n" + "Do not output anything else."


def build_prompt(item: dict, style: str = "paper", reason_max_words: int = 20) -> str:
    prompt = str(item["prompt"]).strip()
    if style == "direct":
        return prompt
    if style != "paper":
        raise ValueError(f"unknown prompt style: {style}")
    if item["capability_code"] == "Gf":
        return _reasoning_wrapper(prompt)
    return _capped_wrapper(prompt, allowed_labels(item), reason_max_words)
