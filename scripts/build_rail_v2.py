#!/usr/bin/env python3
"""Rebuild the RAIL benchmark from its v1 release with the paper's taxonomy.

Taxonomy follows arXiv:2606.11260 (Table 1 and Appendix "CHC-Aligned Task Design Examples"):
5 core capabilities -> 32 sub-capabilities (tasks) -> optional subtasks.

Input : <v1>/data/all.jsonl and <v1>/audio (the v1 release, commit 8ef268a of the GitHub and
        Hugging Face repositories), plus sources/ in this folder
Output: <out>/data/<capability>/<TASK>.jsonl, <out>/audio/<capability>/<TASK>/..., README.md
        (Hugging Face dataset card), metadata.json (Croissant), build_report.json.
        <out> defaults to this script's folder.

Memory prompts are rebuilt with the same logic as the evaluation-time builder
that produced the paper's memory prompts, because
v1 dropped the options from several memory prompts. UM speaker-tracking items are taken from
the evaluated manifest in sources/ (see UM_SPEAKER_TRACKING_EVAL).

Run with: python build_rail_v2.py --v1-root /path/to/rail-v1 [--out /path/to/release]
Needs soundfile for audio durations.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import statistics
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import soundfile as sf
except ImportError:  # durations fall back to v1 values
    sf = None

SCRIPT_DIR = Path(__file__).resolve().parent
V2_ROOT = SCRIPT_DIR  # output folder; override with --out
V1_ROOT = Path(os.environ.get("RAIL_V1_ROOT", SCRIPT_DIR.parent / "RAIL"))
DATA_DIR = V2_ROOT / "data"
AUDIO_DIR = V2_ROOT / "audio"

HF_REPO = "AIMS-RAIL/RAIL"
HF_URL = f"https://huggingface.co/datasets/{HF_REPO}"
ARXIV_ID = "2606.11260"
PAPER_TITLE = (
    "RAIL: Rethinking Auditory Intelligence in Large Audio-Language Models "
    "with a CHC-Grounded Benchmark"
)
VERSION = "2.0.0"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# ---------------------------------------------------------------------------
# Taxonomy (paper order). Each task: (code, paper name, v1 `ability` label)
# ---------------------------------------------------------------------------
TAXONOMY = [
    {
        "key": "ga",
        "code": "Ga",
        "name": "Auditory Processing",
        "slug": "auditory_processing",
        "tasks": [
            ("US", "Speech Sound Discrimination", "Speech Sound Discrimination (US)"),
            ("UR", "Resistance to Auditory Stimulus Distortion", "Resistance Auditory Stimulus Distortion (UR)"),
            ("U8", "Maintaining and Judging Rhythm", "Maintaining and Judging Rhythm (U8)"),
            ("UP", "Absolute Pitch", "Absolute Pitch (UP)"),
            ("U1/U9", "Musical Discrimination and Judgment", "Musical Discrimination and Judgment (U1 U9)"),
            ("UL", "Sound Localization", "Sound Localization (UL)"),
            ("PC", "Phonetic Coding", "Phonetic Coding (PC)"),
        ],
    },
    {
        "key": "gf",
        "code": "Gf",
        "name": "Reasoning",
        "slug": "reasoning",
        "tasks": [
            ("I", "Induction", "Induction (I)"),
            ("RG", "General Sequential Reasoning", "General Sequential Reasoning (RG)"),
            ("RQ", "Quantitative Reasoning", "Quantitative Reasoning (RQ)"),
        ],
    },
    {
        "key": "gsm",
        "code": "Gsm/Glr",
        "name": "Memory",
        "slug": "memory",
        "tasks": [
            ("MS", "Memory Span", "Memory Span (MS)"),
            ("MA", "Associative Memory", "Associative Memory (MA)"),
            ("MM", "Meaningful Memory", "Meaningful Memory (MM)"),
            ("M6", "Free Recall Memory", "Free-Recall Memory (M6)"),
            ("UM", "Memory for Sound Patterns", "Memory for Sound Patterns (UM)"),
            ("WM", "Working Memory", "Working Memory (WM)"),
        ],
    },
    {
        "key": "gs",
        "code": "Gs",
        "name": "Processing Efficiency",
        "slug": "processing_efficiency",
        "tasks": [
            ("P", "Perceptual Speed", "Perceptual Speed (P)"),
            ("R9", "Rate-of-Test-Taking", "Rate-of-Test-Taking (R9)"),
            ("N", "Number Facility", "Number Facility (N)"),
            ("RS", "Reading Speed", "Reading Speed (RS)"),
            ("R1", "Simple Reaction Time", "Simple Reaction Time (R1)"),
            ("R2", "Choice Reaction Time", "Choice Reaction Time (R2)"),
            ("R4", "Semantic Processing Speed", "Semantic Processing Speed (R4)"),
            ("R7", "Mental Comparison Speed", "Mental Comparison Speed (R7)"),
            ("IT", "Inspection Time", "Inspection Time (IT)"),
        ],
    },
    {
        "key": "gc",
        "code": "Gc/Gkn",
        "name": "Knowledge",
        "slug": "knowledge",
        "tasks": [
            ("K0", "General (Verbal) Information", "General (verbal) Information (K0)"),
            ("LD", "Language Development", "Language Development (LD)"),
            ("LS", "Listening Ability", "Listening Ability (LS)"),
            ("KL", "Foreign Language Proficiency", "Foreign Language Proficiency (KL)"),
            ("A5", "Geography Achievement", "Geography Achievement (A5)"),
            ("MK", "Mechanical Knowledge", "Mechanical Knowledge (MK)"),
            ("BC", "Knowledge of Behavioral Content", "Knowledge Behavioral Content (BC)"),
        ],
    },
]

# Table 1 of the paper, used as a build-time check.
PAPER_STATS = {
    "Ga": {"samples": 1170, "tasks": 7, "total_h": 4.93, "mean_s": 15.17, "median_s": 4.00},
    "Gf": {"samples": 322, "tasks": 3, "total_h": 5.19, "mean_s": 39.04, "median_s": 25.04},
    "Gsm/Glr": {"samples": 1000, "tasks": 6, "total_h": 13.0, "mean_s": 46.65, "median_s": 28.64},
    "Gs": {"samples": 1800, "tasks": 9, "total_h": 2.01, "mean_s": 4.02, "median_s": 1.81},
    "Gc/Gkn": {"samples": 1014, "tasks": 7, "total_h": 7.2, "mean_s": 25.48, "median_s": 10.49},
}

ABILITY_TO_TASK: Dict[str, Tuple[dict, str, str]] = {}
for _cap in TAXONOMY:
    for _code, _name, _legacy in _cap["tasks"]:
        ABILITY_TO_TASK[_legacy] = (_cap, _code, _name)

# v1 gold answers that are not verbatim among the options. Only near-exact matches are
# resolved; the rest are reported in build_report.json and keep answer_label = null.
GOLD_OPTION_FIXES = {
    "neutralisation reaction": "neutralization reaction",
    "It occurs in the beginning": "It occurs only once in the beginning",
}

# Evaluation-time prompt constants (from the paper's memory manifest builder).
UM_PROSODIC_EVAL_PROMPT = (
    "You will hear one template prosodic pattern followed by four candidate segments. "
    "Candidate 1 maps to A, Candidate 2 maps to B, Candidate 3 maps to C, and Candidate 4 maps to D. "
    "Which candidate matches the template pattern?"
)
MM_MC_LISTEN_PROMPT = "Listen to the entire audio carefully."
MM_RACE_OPTIONFIX_REPLACEMENT = (
    "Not enough related information is provided to identify the direct prior cause."
)
MM_RACE_SUBSET = "race_relation_original"
MM_QUOTED_PHRASE_RE = re.compile(r'"([^"]+)"')
MS_QUESTION = "Listen to the sequence. Choose the option with the exact same order."
RETURN_LETTER = "Return ONLY one letter."

# UM speaker-tracking items as evaluated in the paper (run26 manifest, 2026-04-22). Same audio as v1,
# but v1 carries an earlier question wording (and, for some items, a different question).
UM_SPEAKER_TRACKING_EVAL = {
    r["item_id"]: r
    for r in map(json.loads, (SCRIPT_DIR / "sources/um_speaker_tracking_eval_20260422.jsonl").open(encoding="utf-8"))
}

PATH_LIKE_KEYS = {
    "audio_path",
    "audio_paths",
    "audio_file",
    "resolved_audio",
    "original_audio_paths",
    "source_audio_path",
    "source_audio_file",
}

# ---------------------------------------------------------------------------
# Subtasks. Reasoning subtasks are the paper's (Table: PA/RI, Cnt/Math, CP/OS);
# the others follow the "LLM Task Design" column of the paper's appendix and are
# read from the item's source directory or question template.
# ---------------------------------------------------------------------------


def _dir(row: dict) -> str:
    return row["audio"].lower()


def subtask_for(code: str, row: dict, meta: dict) -> Optional[str]:
    q = (row.get("question") or "").lower()
    if code == "I":
        return {"pattern_abstraction": "Pattern Abstraction", "rule_induction": "Rule Induction"}[
            meta["sub_task"]
        ]
    if code == "RG":
        if row["subset"] == "cognitive_puzzles":
            return "Auditory Cognitive Puzzle"
        return "Sequential Reasoning with General Rules"
    if code == "RQ":
        return "Math Reasoning" if meta.get("_dataset") == "Spoken_MQA" else "Counting"
    if code == "U8":
        for key, name in (
            ("/beatregularity/", "Beat Regularity Detection"),
            ("/meteridentification/", "Meter Identification"),
            ("/tempochange/", "Tempo Change Detection"),
            ("/tempopairs/", "Tempo Comparison"),
        ):
            if key in _dir(row):
                return name
    if code == "UR":
        for key, name in (
            ("/babble/", "Babble Noise"),
            ("/reverb/", "Reverberation"),
            ("/music/", "Background Music"),
            ("/bandlimit/", "Band-Limiting"),
        ):
            if key in _dir(row):
                return name
    if code == "U1/U9":
        for key, name in (
            ("/instrument/", "Instrument Identification"),
            ("/genre/", "Genre Recognition"),
            ("/musical/", "Musical Structure Analysis"),
            ("/aesthetic/", "Aesthetic and Emotional Judgment"),
        ):
            if key in _dir(row):
                return name
    if code == "PC":
        if "left-channel" in q:
            return "Stereo-Separated Phoneme Identification"
        if "begin at slightly" in q:
            return "Staggered-Onset Phoneme Identification"
        if "which two phonemes" in q:
            return "Concurrent Phoneme Segregation"
        if "which phoneme do you hear" in q:
            return "Isolated Phoneme Identification"
    if code == "UP":
        if "midi pitch" in q:
            return "MIDI Pitch Identification"
        if "what note is being played" in q:
            return "Note Name Identification"
        if "higher or lower" in q:
            return "Relative Pitch Comparison"
        if "frequency" in q:
            return "Frequency Estimation"
    if code == "US":
        if "same or different" in q:
            return "Minimal-Pair Discrimination"
        if "what word did you hear" in q:
            return "Confusable Word Identification"
        if "sounds more" in q:
            return "Emotional Prosody Comparison"
    if code == "UL":
        if "azimuth" in q:
            return "Azimuth Estimation"
        if "how far away" in q:
            return "Distance Estimation"
        if any(k in q for k in ("trajectory", "motion path", "movement")):
            return "Motion Trajectory"
        return "Direction Identification"
    if code == "MK":
        if "operating normally" in q:
            return "Anomaly Detection"
        if "what machine" in q:
            return "Machine Identification"
        if "function" in q:
            return "Machine Function Inference"
    if code == "UM":
        return {
            "nback": "N-Back",
            "speaker_tracking": "Speaker Tracking",
            "prosodic_matching": "Prosodic Contour Matching",
        }[meta["subtask"]]
    if code == "MM":
        return "Narrated Passage" if meta.get("subset") == MM_RACE_SUBSET else "Multi-Turn Dialogue"
    if code in {"US", "UR", "U8", "UP", "U1/U9", "PC", "MK"}:
        raise ValueError(f"no subtask rule matched for {code}: {row['id']} {q[:80]!r}")
    return None


# ---------------------------------------------------------------------------
# Field normalisation
# ---------------------------------------------------------------------------

LETTER_PREFIX_RE = re.compile(r"^\s*([A-Z])[\.\)]\s+")


def strip_letter_prefixes(choices: List[str]) -> List[str]:
    """Remove 'A. ' prefixes only when every option carries its own sequential letter."""
    if choices and all(
        (m := LETTER_PREFIX_RE.match(c)) and m.group(1) == LETTERS[i] for i, c in enumerate(choices)
    ):
        return [LETTER_PREFIX_RE.sub("", c, count=1).strip() for c in choices]
    return [c.strip() for c in choices]


def strip_choices_from_question(question: str, choices: List[str]) -> str:
    """Auditory/Knowledge prompts inline the options ('... Choices: A. x B. y'); keep the stem."""
    text = question
    for marker in (" Choices:", "\nChoices:", "Choices:", " Options:", "\nOptions:"):
        idx = text.find(marker)
        if idx > 0:
            return text[:idx].strip()
    if choices:
        first = f"A. {choices[0]}"
        idx = text.find(first)
        if idx > 0:
            return text[:idx].strip()
    return text.strip()


def lettered(choices: List[str]) -> List[str]:
    return [f"{LETTERS[i]}. {c}" for i, c in enumerate(choices)]


def mcq_from_v1(row: dict, issues: List[dict]) -> Tuple[List[str], Optional[str], Optional[str]]:
    choices = strip_letter_prefixes(row["choices"])
    label = row.get("answer_label")
    answer = row.get("answer")
    answer_text = LETTER_PREFIX_RE.sub("", answer, count=1).strip() if answer else answer
    if label and label in LETTERS[: len(choices)]:
        gold = choices[LETTERS.index(label)]
        if answer_text != gold:
            issues.append({"id": row["id"], "issue": "answer_text_differs_from_labelled_option",
                           "v1_answer": answer, "option": gold})
        return choices, gold, label
    fixed = GOLD_OPTION_FIXES.get(answer_text or "")
    target = fixed if fixed else answer_text
    if target in choices:
        idx = choices.index(target)
        if fixed:
            issues.append({"id": row["id"], "issue": "gold_fixed_to_matching_option",
                           "v1_answer": answer, "option": target, "label": LETTERS[idx]})
        return choices, target, LETTERS[idx]
    issues.append({"id": row["id"], "issue": "gold_not_in_choices", "v1_answer": answer,
                   "choices": choices})
    return choices, answer_text, None


def build_memory(code: str, row: dict, meta: dict) -> Dict[str, Any]:
    """Mirror of the paper's memory manifest builders (model-facing prompt + gold)."""
    if code == "MS":
        choices = [", ".join(str(x) for x in opt) for opt in meta["mc_options"]]
        label = str(meta["mc_answer_label"])
        prompt = MS_QUESTION + "\n" + "\n".join(lettered(choices)) + "\n" + RETURN_LETTER
        return dict(question=MS_QUESTION, choices=choices, label=label, prompt=prompt, fmt="multiple_choice")
    if code == "MA":
        choices = [str(x) for x in meta["mc_options"]]
        question = str(meta.get("query", "")).strip()
        prompt = question + "\n" + "\n".join(lettered(choices)) + "\n" + RETURN_LETTER
        return dict(question=question, choices=choices, label=str(meta["mc_answer_label"]), prompt=prompt,
                    fmt="multiple_choice")
    if code == "WM":
        choices = [", ".join(str(x) for x in opt) for opt in meta["mc_options"]]
        question = "\n".join(
            p for p in (str(meta.get("instructions", "")).strip(), str(meta.get("question", "")).strip()) if p
        ) or "Listen to the audio and choose the option that matches the final list exactly."
        prompt = question + "\n" + "\n".join(lettered(choices)) + "\n" + RETURN_LETTER
        return dict(question=question, choices=choices, label=str(meta["mc_answer_label"]), prompt=prompt,
                    fmt="multiple_choice")
    if code == "MM":
        choices = [str(x) for x in meta["mc_options"]]
        label = str(meta["mc_answer_label"])
        question = str(meta.get("question", "")).strip()
        if str(meta.get("subset", "")).strip().lower() == MM_RACE_SUBSET:
            match = MM_QUOTED_PHRASE_RE.search(question)
            if match:
                phrase = match.group(1).strip().lower()
                for i, opt in enumerate(choices):
                    if phrase and phrase in opt.lower():
                        if LETTERS[i] != label:
                            choices[i] = MM_RACE_OPTIONFIX_REPLACEMENT
                        break
        instructions = str(meta.get("instructions", "") or meta.get("instruction", "")).strip()
        instructions = instructions or MM_MC_LISTEN_PROMPT
        prompt = "\n".join(p for p in (instructions, question) if p) + "\n" + "\n".join(lettered(choices)) + \
            "\n" + RETURN_LETTER
        return dict(question=question, choices=choices, label=label, prompt=prompt, fmt="multiple_choice")
    if code == "M6":
        facts = [str(x).strip() for x in meta.get("target_facts", []) if str(x).strip()]
        prompt = str(meta.get("prompt") or meta.get("question") or "").strip()
        return dict(question=prompt, choices=[], label=None, prompt=prompt, fmt="free_recall",
                    answer="\n".join(facts))
    if code == "UM":
        sub = meta["subtask"]
        if sub == "nback":
            question = str(meta.get("question", "")).strip()
            return dict(question=question, choices=["same", "different"], label=None,
                        prompt=question + "\nReturn ONLY one word: same or different.", fmt="closed_label",
                        answer=str(meta["answer"]))
        if sub == "speaker_tracking":
            evaluated = UM_SPEAKER_TRACKING_EVAL.get(f"um_{meta['id']}")
            if evaluated:
                gold = evaluated["gold_payload"]
                choices = [str(x) for x in gold["mc_options"]]
                prompt = evaluated["prompt"]
                question = prompt.split("\nA. ", 1)[0].strip()
                if prompt != question + "\n" + "\n".join(lettered(choices)) + "\n" + RETURN_LETTER:
                    raise ValueError(f"unexpected evaluated UM prompt layout: um_{meta['id']}")
                return dict(question=question, choices=choices, label=str(evaluated["answer"]), prompt=prompt,
                            fmt="multiple_choice")
            choices = [str(x) for x in meta["mc_options"]]
            question = str(meta.get("question", "")).strip()
            prompt = question + "\n" + "\n".join(lettered(choices)) + "\n" + RETURN_LETTER
            return dict(question=question, choices=choices, label=str(meta["mc_answer_label"]), prompt=prompt,
                        fmt="multiple_choice")
        ordinal = {"A": "First candidate segment", "B": "Second candidate segment",
                   "C": "Third candidate segment", "D": "Fourth candidate segment"}
        options = meta["options"]
        if isinstance(options, str):
            options = json.loads(options)
        labels = [str(o.get("label", "")).strip().upper() for o in options if str(o.get("label", "")).strip()]
        choices = [ordinal.get(lb, "Candidate segment") for lb in labels]
        prompt = UM_PROSODIC_EVAL_PROMPT + "\n" + "\n".join(f"{lb}. {c}" for lb, c in zip(labels, choices)) + \
            "\n" + RETURN_LETTER
        return dict(question=UM_PROSODIC_EVAL_PROMPT, choices=choices, label=str(meta["mc_answer_label"]),
                    prompt=prompt, fmt="multiple_choice")
    raise KeyError(code)


def normalise(row: dict, meta: dict, cap: dict, code: str, issues: List[dict]) -> Dict[str, Any]:
    """Return question/choices/answer/answer_label/prompt/answer_format for one v1 row."""
    if cap["code"] == "Gsm/Glr":
        out = build_memory(code, row, meta)
        label = out["label"]
        answer = out.get("answer")
        if label is not None:
            answer = out["choices"][LETTERS.index(label)]
        return dict(question=out["question"], choices=out["choices"], answer=answer, answer_label=label,
                    prompt=out["prompt"], answer_format=out["fmt"])

    if cap["code"] == "Gs":
        return dict(question=row["prompt"], choices=[str(x) for x in meta["allowed_outputs"]],
                    answer=str(meta["answer"]), answer_label=None, prompt=row["prompt"],
                    answer_format="closed_label")

    if code == "RQ" and not row["choices"]:
        return dict(question=row["prompt"], choices=[], answer=row["answer"], answer_label=None,
                    prompt=row["prompt"], answer_format="open_ended")

    choices, answer, label = mcq_from_v1(row, issues)
    question = row["question"] or ""
    if cap["code"] in ("Ga", "Gc/Gkn"):
        question = strip_choices_from_question(question, choices)
    return dict(question=question.strip(), choices=choices, answer=answer, answer_label=label,
                prompt=row["prompt"], answer_format="multiple_choice")


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------


def link_or_copy(src: Path, dest: Path, stats: Counter) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if os.path.samefile(src, dest) or dest.stat().st_size == src.stat().st_size:
            stats["existing"] += 1
            return
        dest.unlink()
    try:
        os.link(src, dest)
        stats["hardlink"] += 1
    except OSError:
        shutil.copy2(src, dest)
        stats["copy"] += 1


_duration_cache: Dict[str, float] = {}


def audio_duration(path: Path) -> Optional[float]:
    key = str(path)
    if key not in _duration_cache:
        if sf is None:
            return None
        try:
            _duration_cache[key] = float(sf.info(key).duration)
        except Exception:
            return None
    return _duration_cache[key]


def remap_paths(value: Any, mapping: Dict[str, str]) -> Any:
    if isinstance(value, list):
        return [remap_paths(v, mapping) for v in value]
    if isinstance(value, str):
        return mapping.get(value)
    return value


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def task_config_name(code: str) -> str:
    return code.replace("/", "_")


# Explicit schema for the dataset card: without it, files whose column is all-empty (e.g. M6
# `choices`, Gs `answer_label`) are inferred as null and multi-file configs fail to cast.
CARD_FEATURES = [
    ("id", "string"), ("capability", "string"), ("capability_code", "string"), ("task", "string"),
    ("task_code", "string"), ("subtask", "string"), ("answer_format", "string"), ("question", "string"),
    ("choices", ["string"]), ("answer", "string"), ("answer_label", "string"), ("prompt", "string"),
    ("audio", "string"), ("audio_paths", ["string"]), ("num_audio", "int64"), ("duration_s", "float64"),
    ("source_id", "string"), ("legacy_id", "string"), ("legacy_config", "string"), ("metadata_json", "string"),
]


def card_dataset_info(config_name: str, num_rows: int) -> List[str]:
    lines = [f"- config_name: {config_name}", "  features:"]
    for name, dtype in CARD_FEATURES:
        lines.append(f"  - name: {name}")
        if isinstance(dtype, list):
            lines.append(f"    list: {dtype[0]}")
        else:
            lines.append(f"    dtype: {dtype}")
    lines += ["  splits:", "  - name: test", f"    num_examples: {num_rows}"]
    return lines


def dur_stats(values: List[float]) -> Dict[str, float]:
    return {
        "total_h": round(sum(values) / 3600, 2),
        "mean_s": round(statistics.mean(values), 2),
        "median_s": round(statistics.median(values), 2),
    }


def main() -> None:
    v1_rows = [json.loads(line) for line in (V1_ROOT / "data/all.jsonl").open(encoding="utf-8")]
    issues: List[dict] = []
    audio_stats: Counter = Counter()
    per_task: "OrderedDict[str, List[dict]]" = OrderedDict()
    for cap in TAXONOMY:
        for code, _, _ in cap["tasks"]:
            per_task[code] = []

    for row in v1_rows:
        cap, code, _ = ABILITY_TO_TASK[row["ability"]]
        per_task[code].append(row)

    out_rows: List[dict] = []
    task_meta: Dict[str, dict] = {}
    for cap in TAXONOMY:
        for code, name, _ in cap["tasks"]:
            rows = per_task[code]
            code_slug = task_config_name(code).lower().replace("_", "")
            task_rows = []
            for n, row in enumerate(rows, start=1):
                meta = json.loads(row["metadata_json"])
                new_id = f"{cap['key']}-{code_slug}-{n:04d}"
                fields = normalise(row, meta, cap, code, issues)
                subtask = subtask_for(code, row, meta)

                mapping: Dict[str, str] = {}
                new_paths: List[str] = []
                durations: List[float] = []
                multi = len(row["audio_paths"]) > 1
                for k, old in enumerate(row["audio_paths"], start=1):
                    src = V1_ROOT / old
                    ext = src.suffix.lower()
                    rel = f"audio/{cap['slug']}/{task_config_name(code)}/{new_id}{f'_{k}' if multi else ''}{ext}"
                    link_or_copy(src, V2_ROOT / rel, audio_stats)
                    mapping[old] = rel
                    new_paths.append(rel)
                    d = audio_duration(src)
                    if d is not None:
                        durations.append(d)
                duration = round(sum(durations), 3) if len(durations) == len(new_paths) else row["duration_s"]

                clean_meta = {
                    k: (remap_paths(v, mapping) if k in PATH_LIKE_KEYS else v) for k, v in meta.items()
                }
                record = OrderedDict(
                    id=new_id,
                    capability=cap["name"],
                    capability_code=cap["code"],
                    task=name,
                    task_code=code,
                    subtask=subtask,
                    answer_format=fields["answer_format"],
                    question=fields["question"],
                    choices=fields["choices"],
                    answer=fields["answer"],
                    answer_label=fields["answer_label"],
                    prompt=fields["prompt"],
                    audio=new_paths[0],
                    audio_paths=new_paths,
                    num_audio=len(new_paths),
                    duration_s=duration,
                    source_id=str(row["source_id"]),
                    legacy_id=row["id"],
                    legacy_config=f"{row['benchmark']}__{row['subset']}",
                    metadata_json=json.dumps(clean_meta, ensure_ascii=False, sort_keys=True,
                                             separators=(",", ":")),
                )
                task_rows.append(record)
            path = DATA_DIR / cap["slug"] / f"{task_config_name(code)}.jsonl"
            write_jsonl(path, task_rows)
            task_meta[code] = {"capability": cap["code"], "task": name, "file": path.relative_to(V2_ROOT).as_posix(),
                               "rows": len(task_rows)}
            out_rows.extend(task_rows)

    report = validate(out_rows, issues, audio_stats)
    write_readme(out_rows, report)
    write_croissant(out_rows, task_meta)
    (V2_ROOT / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                                encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("total_rows", "unique_ids", "audio", "checks")}, indent=2))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(rows: List[dict], issues: List[dict], audio_stats: Counter) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    ids = [r["id"] for r in rows]
    checks["ids_unique"] = len(ids) == len(set(ids))
    checks["audio_exists"] = all((V2_ROOT / p).exists() for r in rows for p in r["audio_paths"])
    checks["prompt_present"] = all(r["prompt"] and r["question"] for r in rows)
    checks["answer_present"] = all(r["answer"] not in (None, "") for r in rows)
    mcq = [r for r in rows if r["answer_format"] == "multiple_choice"]
    checks["mcq_label_matches_answer"] = sum(
        1 for r in mcq if r["answer_label"] and r["choices"][LETTERS.index(r["answer_label"])] == r["answer"]
    )
    checks["mcq_total"] = len(mcq)
    checks["closed_label_answer_in_choices"] = all(
        r["answer"] in r["choices"] for r in rows if r["answer_format"] == "closed_label"
    )
    checks["mcq_prompt_contains_all_options"] = sum(
        1 for r in mcq if all(c in r["prompt"] for c in r["choices"])
    )

    per_cap: Dict[str, Any] = OrderedDict()
    for cap in TAXONOMY:
        cap_rows = [r for r in rows if r["capability_code"] == cap["code"]]
        durs = [r["duration_s"] for r in cap_rows if r["duration_s"] is not None]
        tasks = OrderedDict()
        for code, name, _ in cap["tasks"]:
            t_rows = [r for r in cap_rows if r["task_code"] == code]
            tasks[code] = {
                "task": name,
                "rows": len(t_rows),
                "answer_format": dict(Counter(r["answer_format"] for r in t_rows)),
                "subtasks": dict(Counter(r["subtask"] for r in t_rows if r["subtask"])),
            }
        per_cap[cap["code"]] = {
            "capability": cap["name"],
            "rows": len(cap_rows),
            "tasks": len(tasks),
            "duration_rows": len(durs),
            **(dur_stats(durs) if durs else {}),
            "paper_table1": PAPER_STATS[cap["code"]],
            "task_breakdown": tasks,
        }
        checks[f"{cap['code']}_rows_match_paper"] = len(cap_rows) == PAPER_STATS[cap["code"]]["samples"]
        checks[f"{cap['code']}_tasks_match_paper"] = len(tasks) == PAPER_STATS[cap["code"]]["tasks"]

    all_durs = [r["duration_s"] for r in rows if r["duration_s"] is not None]
    return {
        "version": VERSION,
        "total_rows": len(rows),
        "unique_ids": len(set(ids)),
        "audio": {"files": sum(len(r["audio_paths"]) for r in rows), **dict(audio_stats)},
        "overall_duration": dur_stats(all_durs),
        "checks": checks,
        "capabilities": per_cap,
        "answer_issues": issues,
    }


# ---------------------------------------------------------------------------
# README (dataset card) and Croissant metadata
# ---------------------------------------------------------------------------


def write_readme(rows: List[dict], report: Dict[str, Any]) -> None:
    yaml = [
        "---",
        "license: cc-by-4.0",
        "language:",
        "- en",
        "pretty_name: RAIL",
        "size_categories:",
        "- 1K<n<10K",
        "task_categories:",
        "- audio-classification",
        "- question-answering",
        "tags:",
        "- audio",
        "- benchmark",
        "- audio-language-models",
        "- cognitive-evaluation",
        "- CHC",
        "configs:",
        "- config_name: all",
        "  default: true",
        "  data_files:",
        "  - split: test",
        "    path: data/*/*.jsonl",
    ]
    for cap in TAXONOMY:
        yaml += [f"- config_name: {cap['slug']}", "  data_files:", "  - split: test",
                 f"    path: data/{cap['slug']}/*.jsonl"]
    for cap in TAXONOMY:
        for code, _, _ in cap["tasks"]:
            cfg = task_config_name(code)
            yaml += [f"- config_name: {cfg}", "  data_files:", "  - split: test",
                     f"    path: data/{cap['slug']}/{cfg}.jsonl"]
    caps = report["capabilities"]
    yaml.append("dataset_info:")
    yaml += card_dataset_info("all", report["total_rows"])
    for cap in TAXONOMY:
        yaml += card_dataset_info(cap["slug"], caps[cap["code"]]["rows"])
    for cap in TAXONOMY:
        for code, t in caps[cap["code"]]["task_breakdown"].items():
            yaml += card_dataset_info(task_config_name(code), t["rows"])
    yaml += ["---", ""]

    # Paper header (title, authors, links, TL;DR) and BibTeX are shared with the GitHub README.
    lines = [
        (SCRIPT_DIR / "sources/paper_header.md").read_text(encoding="utf-8").rstrip(),
        "",
        "## Taxonomy and statistics",
        "",
        "| Capability | Code | Tasks | Samples | Total dur. (h) | Mean dur. (s) | Median dur. (s) | Config |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for cap in TAXONOMY:
        c = caps[cap["code"]]
        lines.append(
            f"| {cap['name']} | {cap['code']} | {c['tasks']} | {c['rows']} | {c.get('total_h', '')} | "
            f"{c.get('mean_s', '')} | {c.get('median_s', '')} | `{cap['slug']}` |"
        )
    od = report["overall_duration"]
    lines += [
        f"| **Overall** | | **32** | **{report['total_rows']}** | {od['total_h']} | {od['mean_s']} | "
        f"{od['median_s']} | `all` |",
        "",
        "Durations are summed over all clips of an item and computed from the audio files.",
        "",
        "### Tasks and subtasks",
        "",
        "Every task is also a config, named by its CHC code (`U1/U9` → `U1_U9`).",
        "",
        "| Capability | Task (code) | Samples | Answer format | Subtasks (samples) |",
        "|---|---|---:|---|---|",
    ]
    for cap in TAXONOMY:
        for code, t in caps[cap["code"]]["task_breakdown"].items():
            fmt = ", ".join(t["answer_format"])
            subs = "; ".join(f"{k} ({v})" for k, v in t["subtasks"].items()) or "—"
            lines.append(f"| {cap['name']} | {t['task']} (`{code}`) | {t['rows']} | {fmt} | {subs} |")
    lines += [
        "",
        "## Loading",
        "",
        "```python",
        "from datasets import load_dataset",
        "from huggingface_hub import snapshot_download",
        "",
        f'root = snapshot_download("{HF_REPO}", repo_type="dataset")  # audio + manifests',
        f'ds = load_dataset("{HF_REPO}", "all", split="test")        # or "memory", "UL", "U1_U9", ...',
        "",
        "item = ds[0]",
        'audio_files = [f"{root}/{p}" for p in item["audio_paths"]]',
        'print(item["prompt"])',
        "```",
        "",
        "## Fields",
        "",
        "| Field | Description |",
        "|---|---|",
        "| `id` | Unique item id, `<capability>-<task>-<nnnn>` (e.g. `ga-ul-0001`). |",
        "| `capability`, `capability_code` | Core capability and CHC broad-ability code (`Ga`, `Gf`, `Gsm/Glr`, `Gs`, `Gc/Gkn`). |",
        "| `task`, `task_code` | Sub-capability (CHC narrow ability) and its code, as named in the paper. |",
        "| `subtask` | Finer task design within a sub-capability; `null` where the task has a single design. |",
        "| `answer_format` | `multiple_choice`, `closed_label` (answer is one of `choices`, no letters shown), `open_ended`, or `free_recall`. |",
        "| `question` | Question stem without options or output instructions. |",
        "| `choices` | Option texts without letter prefixes (`multiple_choice`), or the allowed output labels (`closed_label`). |",
        "| `answer` | Gold answer text. For `free_recall`, one target item per line. |",
        "| `answer_label` | Gold option letter for `multiple_choice`; `null` otherwise. |",
        "| `prompt` | Full model-facing prompt used in the paper's evaluation (question, options and output instruction). |",
        "| `audio`, `audio_paths`, `num_audio` | Primary clip, all clips in presentation order, and clip count (paths are relative to the repo root). |",
        "| `duration_s` | Total audio duration of the item in seconds. |",
        "| `source_id` | Item id in the source manifest the item was generated from. |",
        "| `legacy_id`, `legacy_config` | Id and config name in RAIL v1, for mapping earlier results. |",
        "| `metadata_json` | Original source row as JSON text (generation parameters, conditions, provenance). |",
        "",
        "## Evaluation",
        "",
        "- **Multiple choice / closed label**: a prediction is correct when it contains every token of the gold "
        "answer (or its letter) and no token unique to another option (tokenizer `[A-Za-z0-9]+`), as defined in the paper. "
        "An LLM-as-judge score is reported alongside.",
        "- **Processing Efficiency (Gs)**: models answer in the form `Reason: ...; Answer: ...`; accuracy is combined "
        "with the reason length into the budgeted accuracy curve and its normalised area (B-AUC, budgets 0–50 tokens).",
        "- **Free recall (M6)**: recall of the target items listed in `answer`.",
        "- **Open-ended math (RQ, Math Reasoning)**: the final numeric answer is compared with `answer`.",
        "",
        "## Changes from v1",
        "",
        "- Labels follow the paper: `capability` → `task` (32) → `subtask`; v1 `benchmark`/`subset`/`task`/`ability` are replaced.",
        "- Configs are organised by capability and by task instead of by source manifest.",
        "- Ids are unique (v1 had 35 duplicated ids); v1 ids are kept in `legacy_id`.",
        "- Memory prompts and options are the exact ones used in evaluation (v1 lacked prompts for Memory Span, options for Working/Meaningful Memory prompts, and options for prosodic matching).",
        "- Speaker-tracking items (Memory for Sound Patterns) use the evaluated question wording and options; v1 carried an earlier wording for the same audio.",
        "- Options no longer carry letter prefixes; efficiency tasks list their allowed output labels.",
        "- `duration_s` is filled for every item from the audio files.",
        "- Audio is stored as `audio/<capability>/<task>/<id>[_k].<ext>`.",
        "",
        "## Citation",
        "",
        (SCRIPT_DIR / "sources/citation.md").read_text(encoding="utf-8").rstrip(),
        "",
        "## License",
        "",
        "CC-BY-4.0. Some items are derived from existing corpora; use them within the terms of those sources.",
        "",
    ]
    (V2_ROOT / "README.md").write_text("\n".join(yaml + lines), encoding="utf-8")


def write_croissant(rows: List[dict], task_meta: Dict[str, dict]) -> None:
    v1 = json.loads((V1_ROOT / "metadata.json").read_text(encoding="utf-8"))
    field_desc = [
        ("id", "sc:Text", "Unique item id."),
        ("capability", "sc:Text", "CHC core capability."),
        ("capability_code", "sc:Text", "CHC broad-ability code."),
        ("task", "sc:Text", "Sub-capability (CHC narrow ability)."),
        ("task_code", "sc:Text", "CHC narrow-ability code."),
        ("subtask", "sc:Text", "Task design within the sub-capability, if any."),
        ("answer_format", "sc:Text", "multiple_choice, closed_label, open_ended or free_recall."),
        ("question", "sc:Text", "Question stem."),
        ("choices", "sc:Text", "Option texts or allowed output labels."),
        ("answer", "sc:Text", "Gold answer text."),
        ("answer_label", "sc:Text", "Gold option letter for multiple-choice items."),
        ("prompt", "sc:Text", "Full model-facing prompt used in evaluation."),
        ("audio", "sc:Text", "Primary relative audio path."),
        ("audio_paths", "sc:Text", "All relative audio paths in presentation order."),
        ("num_audio", "sc:Integer", "Number of audio clips."),
        ("duration_s", "sc:Float", "Total audio duration in seconds."),
        ("source_id", "sc:Text", "Item id in the source manifest."),
        ("legacy_id", "sc:Text", "Item id in RAIL v1."),
        ("legacy_config", "sc:Text", "Config name in RAIL v1."),
        ("metadata_json", "sc:Text", "Original source row as JSON text."),
    ]
    distribution = [
        {
            "@type": "cr:FileSet",
            "@id": "manifests",
            "name": "manifests",
            "description": "Per-task JSONL manifests, one JSON object per item.",
            "contentUrl": f"{HF_URL}/tree/main/data",
            "encodingFormat": "application/jsonlines",
            "includes": "data/*/*.jsonl",
        },
        {
            "@type": "cr:FileSet",
            "@id": "audio_files",
            "name": "audio",
            "description": "Audio files referenced by relative paths in the manifests.",
            "contentUrl": f"{HF_URL}/tree/main/audio",
            "encodingFormat": ["audio/wav", "audio/mpeg", "audio/flac"],
            "includes": "audio/**/*",
        },
    ]
    for code, t in task_meta.items():
        distribution.append({
            "@type": "cr:FileObject",
            "@id": f"task_{task_config_name(code)}",
            "name": Path(t["file"]).name,
            "description": f"{t['task']} ({code}), {t['capability']}.",
            "contentUrl": f"{HF_URL}/resolve/main/{t['file']}",
            "encodingFormat": "application/jsonlines",
            "sha256": sha256_file(V2_ROOT / t["file"]),
            "containedIn": {"@id": "manifests"},
        })
    croissant = {
        "@context": v1["@context"],
        "@type": "sc:Dataset",
        "name": "RAIL",
        "description": (
            "RAIL is a CHC-grounded benchmark for large audio-language models with 5,306 items over "
            "5 core capabilities (Auditory Processing, Reasoning, Memory, Processing Efficiency, Knowledge) "
            "and 32 sub-capabilities, covering speech, environmental sound and music."
        ),
        "url": HF_URL,
        "license": "https://spdx.org/licenses/CC-BY-4.0.html",
        "version": VERSION,
        "datePublished": "2026-06-09",
        "citeAs": f"{PAPER_TITLE}. arXiv:{ARXIV_ID}, 2026.",
        "conformsTo": "http://mlcommons.org/croissant/1.1",
        "distribution": distribution,
        "recordSet": [{
            "@type": "cr:RecordSet",
            "@id": "rail_records",
            "name": "rail_records",
            "description": "Item-level records of RAIL.",
            "field": [
                {
                    "@type": "cr:Field",
                    "@id": f"rail_records/{name}",
                    "name": name,
                    "description": desc,
                    "dataType": dtype,
                    **({"repeated": True} if name in ("choices", "audio_paths") else {}),
                    "source": {"fileSet": {"@id": "manifests"}, "extract": {"column": name}},
                }
                for name, dtype, desc in field_desc
            ],
        }],
    }
    for key, value in v1.items():
        if key.startswith("rai:"):
            croissant[key] = value
    croissant["prov:wasDerivedFrom"] = [f"{HF_URL} (v1.0.0)"]
    croissant["prov:wasGeneratedBy"] = [{
        "@type": "prov:Activity",
        "name": "RAIL v2 re-organisation",
        "description": (
            "v1 rows were relabelled with the paper's capability/task/subtask taxonomy, given unique ids, "
            "normalised to a single answer schema, and memory prompts were rebuilt as used in evaluation."
        ),
    }]
    (V2_ROOT / "metadata.json").write_text(
        json.dumps(croissant, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v1-root", type=Path, default=V1_ROOT, help="folder of the v1 release")
    parser.add_argument("--out", type=Path, default=V2_ROOT,
                        help="output folder for data/, audio/, README.md (dataset card), metadata.json")
    cli = parser.parse_args()
    V1_ROOT = cli.v1_root.resolve()
    V2_ROOT = cli.out.resolve()
    DATA_DIR, AUDIO_DIR = V2_ROOT / "data", V2_ROOT / "audio"
    main()
