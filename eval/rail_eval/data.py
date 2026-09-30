"""Load RAIL v2 items from a local copy or from the Hugging Face Hub."""

from __future__ import annotations

import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

CAPABILITY_ORDER = ["Ga", "Gf", "Gsm/Glr", "Gs", "Gc/Gkn"]
# Paper order (Appendix "CHC-Aligned Task Design Examples").
TASK_ORDER = [
    "US", "UR", "U8", "UP", "U1/U9", "UL", "PC",
    "I", "RG", "RQ",
    "MS", "MA", "MM", "M6", "UM", "WM",
    "P", "R9", "N", "RS", "R1", "R2", "R4", "R7", "IT",
    "K0", "LD", "LS", "KL", "A5", "MK", "BC",
]


def resolve_root(data: str) -> Path:
    """`data` is a local RAIL v2 folder or a Hub dataset id (downloaded with snapshot_download)."""
    path = Path(data).expanduser()
    if path.is_dir():
        return path.resolve()
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(data, repo_type="dataset"))


def load_items(
    data: str,
    tasks: Optional[Iterable[str]] = None,
    capabilities: Optional[Iterable[str]] = None,
    limit_per_task: int = 0,
    seed: int = 0,
) -> List[dict]:
    """Return items in benchmark order, with `audio_files` resolved to absolute paths.

    tasks / capabilities filter by `task_code` (e.g. "UL", "U1/U9" or "U1_U9") and
    `capability_code` or capability name. `limit_per_task` draws a seeded random subset
    per task (for smoke tests); 0 keeps everything.
    """
    root = resolve_root(data)
    files = sorted((root / "data").glob("*/*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no manifests under {root / 'data'}")
    rows: List[dict] = []
    for f in files:
        with f.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())

    wanted_tasks = {t.replace("_", "/").upper() for t in tasks} if tasks else None
    wanted_caps = {c.lower() for c in capabilities} if capabilities else None
    by_task: "OrderedDict[str, List[dict]]" = OrderedDict()
    for row in sorted(rows, key=lambda r: (TASK_ORDER.index(r["task_code"]), r["id"])):
        if wanted_tasks and row["task_code"].upper() not in wanted_tasks:
            continue
        if wanted_caps and row["capability_code"].lower() not in wanted_caps and \
                row["capability"].lower() not in wanted_caps:
            continue
        by_task.setdefault(row["task_code"], []).append(row)

    out: List[dict] = []
    rng = random.Random(seed)
    for code, items in by_task.items():
        if limit_per_task and len(items) > limit_per_task:
            keep = set(r["id"] for r in rng.sample(items, limit_per_task))
            items = [r for r in items if r["id"] in keep]
        for r in items:
            r["audio_files"] = [str(root / p) for p in r["audio_paths"]]
        out.extend(items)
    return out


def hf_features():
    """`datasets.Features` for the task files; pass to `load_dataset("json", ...)` so files whose
    `choices` or `answer_label` are all empty are not inferred as null."""
    from datasets import Features, Value

    try:
        from datasets import List as HFList  # datasets >= 4
    except ImportError:
        from datasets import Sequence as HFList

    text = Value("string")
    return Features({
        "id": text, "capability": text, "capability_code": text, "task": text, "task_code": text,
        "subtask": text, "answer_format": text, "question": text, "choices": HFList(text),
        "answer": text, "answer_label": text, "prompt": text, "audio": text,
        "audio_paths": HFList(text), "num_audio": Value("int64"), "duration_s": Value("float64"),
        "source_id": text, "legacy_id": text, "legacy_config": text, "metadata_json": text,
    })


def count_by_task(items: List[dict]) -> Dict[str, int]:
    counts: Dict[str, int] = OrderedDict()
    for r in items:
        counts[r["task_code"]] = counts.get(r["task_code"], 0) + 1
    return counts
