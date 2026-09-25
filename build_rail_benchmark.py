#!/usr/bin/env python3
"""Build the RAIL benchmark folder in a Hugging Face friendly layout."""

from __future__ import annotations

import json
import os
import re
import shutil
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any


RAIL_ROOT = Path(__file__).resolve().parent
DATA_DIR = RAIL_ROOT / "data"
AUDIO_DIR = RAIL_ROOT / "audio"

DATASET_NAME = "RAIL Audio Benchmark"
DATASET_DESCRIPTION = (
    "RAIL is an audio benchmark collection for evaluating auditory, cognitive, "
    "quantitative, induction, knowledge, efficiency, and memory reasoning over "
    "single- and multi-audio tasks."
)
# Replace these before final NeurIPS/OpenReview submission.
DATASET_URL = "https://huggingface.co/datasets/AnnoymousNeurLPSsubmit/RAIL"
DATASET_RESOLVE_URL = "https://huggingface.co/datasets/AnnoymousNeurLPSsubmit/RAIL/resolve/main"
DATASET_LICENSE = "https://spdx.org/licenses/CC-BY-4.0.html"
DATASET_VERSION = "1.0.0"
DATASET_DATE_PUBLISHED = "2026-05-05"
DATASET_CITE_AS = (
    "RAIL Audio Benchmark. Hugging Face dataset: "
    "https://huggingface.co/datasets/AnnoymousNeurLPSsubmit/RAIL"
)

PROJECT_ROOT = Path("/data/projects/punim2758/mHealthUnimelb")
GPFS_ROOT = Path("/data/gpfs/projects/punim2758/mHealthUnimelb")
AUDIOCOG_ROOT = PROJECT_ROOT / "AudioCog"
AUDIOCOG_GPFS_ROOT = GPFS_ROOT / "AudioCog"
AUSTIN_ROOT = PROJECT_ROOT / "austinx/audiocog_mem"

SOURCES = [
    {
        "benchmark": "sequential_reasoning",
        "subset": "general_rule",
        "path": AUDIOCOG_GPFS_ROOT
        / "jiahengd/output_sequential/combined_dataset_general_rule.json",
        "format": "json",
        "base_dir": AUDIOCOG_GPFS_ROOT / "jiahengd/output_sequential",
        "primary_audio_keys": ["resolved_audio"],
    },
    {
        "benchmark": "sequential_reasoning",
        "subset": "cognitive_puzzles",
        "path": AUDIOCOG_GPFS_ROOT
        / "jiahengd/auditory_cognitive_puzzles/cognitive_puzzle_samples.json",
        "format": "json",
        "base_dir": AUDIOCOG_GPFS_ROOT / "jiahengd/auditory_cognitive_puzzles",
        "primary_audio_keys": ["resolved_audio"],
    },
    {
        "benchmark": "quantitative_reasoning",
        "subset": "quantitative_reasoning_prompted",
        "path": AUDIOCOG_GPFS_ROOT
        / "jiahengd/quantitative_reasoning/quantitative_reasoning_prompted.json",
        "format": "json",
        "base_dir": AUDIOCOG_GPFS_ROOT / "jiahengd/quantitative_reasoning",
        "primary_audio_keys": ["audio_path"],
    },
    {
        "benchmark": "induction",
        "subset": "induction_dataset_mcq_update",
        "path": AUDIOCOG_GPFS_ROOT / "jiahengd/induction/induction_dataset_mcq_update.json",
        "format": "json",
        "base_dir": AUDIOCOG_GPFS_ROOT / "jiahengd/induction",
        "primary_audio_keys": ["resolved_audio"],
    },
    {
        "benchmark": "auditory",
        "subset": "auditory_manifest_all",
        "path": AUDIOCOG_GPFS_ROOT / "siyiw/datasets/Auditory/auditory_manifest_all.jsonl",
        "format": "jsonl",
        "base_dir": AUDIOCOG_GPFS_ROOT / "siyiw/datasets/Auditory",
        "primary_audio_keys": ["audio_paths", "audio_path", "resolved_audio"],
    },
    {
        "benchmark": "knowledge",
        "subset": "knowledge_manifest_all",
        "path": AUDIOCOG_GPFS_ROOT / "siyiw/datasets/knowledge/knowledge_manifest_all.jsonl",
        "format": "jsonl",
        "base_dir": AUDIOCOG_GPFS_ROOT / "siyiw/datasets/knowledge",
        "primary_audio_keys": ["audio_paths", "audio_path", "resolved_audio"],
    },
    {
        "benchmark": "efficiency",
        "subset": "presentation_benchmark_9x200_bench1800",
        "path": PROJECT_ROOT
        / "hongyuj1/auditory/data/manifests/presentation_benchmark_9x200_bench1800_reading_from_writing_duration.jsonl",
        "format": "jsonl",
        "base_dir": PROJECT_ROOT,
        "primary_audio_keys": ["audio_paths"],
    },
    {
        "benchmark": "memory",
        "subset": "ma",
        "path": AUSTIN_ROOT / "data/manifest_ma.json",
        "format": "json",
        "base_dir": AUSTIN_ROOT,
        "primary_audio_keys": ["audio_path"],
    },
    {
        "benchmark": "memory",
        "subset": "m6",
        "path": AUSTIN_ROOT / "data/manifest_m6.json",
        "format": "json",
        "base_dir": AUSTIN_ROOT,
        "primary_audio_keys": ["audio_path"],
    },
    {
        "benchmark": "memory",
        "subset": "ms",
        "path": AUSTIN_ROOT / "data/manifest_ms.json",
        "format": "json",
        "base_dir": AUSTIN_ROOT,
        "primary_audio_keys": ["audio_path"],
    },
    {
        "benchmark": "memory",
        "subset": "mw_voice_editing_preview",
        "path": AUSTIN_ROOT / "data/manifest_mw_voice_editing_preview.json",
        "format": "json",
        "base_dir": AUSTIN_ROOT,
        "primary_audio_keys": ["audio_path"],
    },
    {
        "benchmark": "memory",
        "subset": "um",
        "path": AUSTIN_ROOT / "data/manifest_um.json",
        "format": "json",
        "base_dir": AUSTIN_ROOT,
        "primary_audio_keys": ["audio_path"],
    },
    {
        "benchmark": "memory",
        "subset": "mm_combination",
        "path": AUSTIN_ROOT / "mm_combination/data/manifest_mm_combination.json",
        "format": "json",
        "base_dir": AUSTIN_ROOT / "mm_combination",
        "primary_audio_keys": ["audio_path"],
    },
]

PATH_LIKE_KEYS = {
    "audio_path",
    "audio_paths",
    "audio_file",
    "resolved_audio",
    "original_audio_paths",
    "source_audio_path",
    "source_audio_file",
}

REASONING_TASKS = {"quantitative_reasoning", "sequential_reasoning", "induction"}

REASONING_ABILITY_BY_TASK = {
    "induction": "Induction (I)",
    "quantitative_reasoning": "Quantitative Reasoning (RQ)",
    "sequential_reasoning": "General Sequential Reasoning (RG)",
}

PROCESSING_EFFICIENCY_ABILITY_BY_CODE = {
    "IT": "Inspection Time (IT)",
    "N": "Number Facility (N)",
    "P": "Perceptual Speed (P)",
    "R1": "Simple Reaction Time (R1)",
    "R2": "Choice Reaction Time (R2)",
    "R4": "Semantic Processing Speed (R4)",
    "R7": "Mental Comparison Speed (R7)",
    "R9": "Rate-of-Test-Taking (R9)",
    "RS": "Reading Speed (RS)",
}

MEMORY_ABILITY_BY_SUBSET = {
    "m6": "Free-Recall Memory (M6)",
    "ma": "Associative Memory (MA)",
    "mm_combination": "Meaningful Memory (MM)",
    "ms": "Memory Span (MS)",
    "mw_voice_editing_preview": "Working Memory (WM)",
    "um": "Memory for Sound Patterns (UM)",
}

AUDITORY_ABILITY_BY_TASK = {
    "Absolute Pitch": "Absolute Pitch (UP)",
    "Maintaining and Judging Rhythm": "Maintaining and Judging Rhythm (U8)",
    "Musical Discrimination and Judgment": "Musical Discrimination and Judgment (U1 U9)",
    "Phonetic Coding": "Phonetic Coding (PC)",
    "Resistance to Auditory Stimulus Distortion": "Resistance Auditory Stimulus Distortion (UR)",
    "Sound Localization": "Sound Localization (UL)",
    "Speech Sound Discrimination": "Speech Sound Discrimination (US)",
}

KNOWLEDGE_ABILITY_BY_TASK = {
    "Foreign Language Proficiency": "Foreign Language Proficiency (KL)",
    "General (Verbal) Information": "General (verbal) Information (K0)",
    "Geography Achievements": "Geography Achievement (A5)",
    "Knowledge of Behavioral Content": "Knowledge Behavioral Content (BC)",
    "Language Development": "Language Development (LD)",
    "Listening Ability": "Listening Ability (LS)",
    "Mechanical Knowledge": "Mechanical Knowledge (MK)",
}


def slug(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "unknown"


def output_benchmark(source: dict[str, Any]) -> str:
    benchmark = source["benchmark"]
    return "reasoning" if benchmark in REASONING_TASKS else benchmark


def get_ability(row: dict[str, Any], source: dict[str, Any]) -> str:
    benchmark = source["benchmark"]
    subset = source["subset"]

    if benchmark in REASONING_TASKS:
        return REASONING_ABILITY_BY_TASK[benchmark]

    if benchmark == "efficiency":
        code = row.get("ability")
        if code in PROCESSING_EFFICIENCY_ABILITY_BY_CODE:
            return PROCESSING_EFFICIENCY_ABILITY_BY_CODE[code]

    if benchmark == "memory":
        if subset in MEMORY_ABILITY_BY_SUBSET:
            return MEMORY_ABILITY_BY_SUBSET[subset]

    if benchmark == "auditory":
        task_name = row.get("task_name") or row.get("task")
        if task_name in AUDITORY_ABILITY_BY_TASK:
            return AUDITORY_ABILITY_BY_TASK[task_name]

    if benchmark == "knowledge":
        task_name = row.get("task_name") or row.get("task")
        if task_name in KNOWLEDGE_ABILITY_BY_TASK:
            return KNOWLEDGE_ABILITY_BY_TASK[task_name]

    raise ValueError(
        "No normalized ability mapping for "
        f"benchmark={benchmark!r}, subset={subset!r}, "
        f"task={row.get('task')!r}, task_name={row.get('task_name')!r}, "
        f"ability={row.get('ability')!r}"
    )


def load_records(source: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(source["path"])
    if source["format"] == "jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    with path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("data"), list):
        return obj["data"]
    raise ValueError(f"Unsupported JSON root in {path}")


def iter_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalize_legacy_audio_path(path: Path) -> Path:
    text = str(path)
    legacy_prefix = "/home/student.unimelb.edu.au/yxiao9550/audiocog_mem/"
    if text.startswith(legacy_prefix):
        rel = text[len(legacy_prefix) :]
        rel = rel.replace("data/audio_mw_voice_editing_preview/", "data/audio_mw_voice_editing/")
        return AUSTIN_ROOT / rel
    return path


def resolve_audio_path(raw: Any, base_dir: Path) -> Path | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    value = raw.strip()
    if value.startswith("http://") or value.startswith("https://"):
        return None

    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    path = normalize_legacy_audio_path(path)

    # /data/gpfs/projects and /data/projects point to the same project in this workspace.
    if not path.exists() and str(path).startswith(str(GPFS_ROOT)):
        candidate = PROJECT_ROOT / path.relative_to(GPFS_ROOT)
        if candidate.exists():
            path = candidate
    if not path.exists() and str(path).startswith(str(PROJECT_ROOT)):
        candidate = GPFS_ROOT / path.relative_to(PROJECT_ROOT)
        if candidate.exists():
            path = candidate

    return path if path.exists() else None


def source_relative_path(src: Path, base_dir: Path) -> Path:
    anchors = [base_dir, AUDIOCOG_GPFS_ROOT, AUDIOCOG_ROOT, AUSTIN_ROOT, PROJECT_ROOT, GPFS_ROOT]
    for anchor in anchors:
        try:
            return src.relative_to(anchor)
        except ValueError:
            continue
    return Path(src.name)


def safe_dest_path(
    src: Path, source: dict[str, Any], used_relpaths: set[str], src_to_rel: dict[str, str]
) -> str:
    src_key = str(src.resolve())
    if src_key in src_to_rel:
        return src_to_rel[src_key]

    rel = source_relative_path(src, Path(source["base_dir"]))
    dest_rel = Path("audio") / source["benchmark"] / source["subset"] / rel
    dest_rel = Path(*[slug(part) if part != src.name else part for part in dest_rel.parts])
    dest_text = dest_rel.as_posix()
    if dest_text in used_relpaths:
        stem = dest_rel.stem
        suffix = dest_rel.suffix
        parent = dest_rel.parent
        index = 2
        while True:
            candidate = (parent / f"{stem}_{index}{suffix}").as_posix()
            if candidate not in used_relpaths:
                dest_text = candidate
                break
            index += 1
    used_relpaths.add(dest_text)
    src_to_rel[src_key] = dest_text
    return dest_text


def link_or_copy(src: Path, rel_dest: str) -> str:
    dest = RAIL_ROOT / rel_dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if dest.stat().st_size == src.stat().st_size:
            return "existing"
        dest.unlink()
    try:
        os.link(src, dest)
        return "hardlink"
    except OSError:
        try:
            shutil.copy2(src, dest)
        except OSError as exc:
            raise PermissionError(f"Cannot copy audio {src} -> {dest}: {exc}") from exc
        return "copy"


def collect_audio_paths(
    row: dict[str, Any],
    source: dict[str, Any],
    used_relpaths: set[str],
    src_to_rel: dict[str, str],
    copy_stats: Counter,
    missing: list[dict[str, Any]],
    primary_only: bool = True,
) -> list[str]:
    keys = source["primary_audio_keys"] if primary_only else PATH_LIKE_KEYS
    out: list[str] = []
    for key in keys:
        if key not in row:
            continue
        for raw in iter_values(row[key]):
            path = resolve_audio_path(raw, Path(source["base_dir"]))
            if path is None:
                if (
                    primary_only
                    and isinstance(raw, str)
                    and ("/" in raw or "\\" in raw)
                    and key != "audio_file"
                ):
                    missing.append(
                        {
                            "benchmark": source["benchmark"],
                            "subset": source["subset"],
                            "key": key,
                            "path": raw,
                        }
                    )
                continue
            rel_dest = safe_dest_path(path, source, used_relpaths, src_to_rel)
            copy_stats[link_or_copy(path, rel_dest)] += 1
            out.append(rel_dest)
    return out


def sanitize_path_value(value: Any, source: dict[str, Any], src_to_rel: dict[str, str]) -> Any:
    if isinstance(value, list):
        return [sanitize_path_value(v, source, src_to_rel) for v in value]
    if not isinstance(value, str):
        return value
    path = resolve_audio_path(value, Path(source["base_dir"]))
    if path is None:
        return None
    return src_to_rel.get(str(path.resolve()))


def sanitize_metadata(row: dict[str, Any], source: dict[str, Any], src_to_rel: dict[str, str]) -> dict[str, Any]:
    clean = {}
    for key, value in row.items():
        if key in PATH_LIKE_KEYS:
            clean[key] = sanitize_path_value(value, source, src_to_rel)
        else:
            clean[key] = value
    return clean


def stringify_choice(choice: Any) -> str:
    if isinstance(choice, str):
        return choice
    if isinstance(choice, list):
        return " | ".join(str(x) for x in choice)
    return json.dumps(choice, ensure_ascii=False, sort_keys=True)


def get_choices(row: dict[str, Any]) -> list[str]:
    choices = row.get("choices")
    if choices is None:
        choices = row.get("mc_options")
    if choices is None:
        choices = row.get("allowed_outputs")
    if isinstance(choices, dict):
        return [f"{key}. {value}" for key, value in sorted(choices.items())]
    if isinstance(choices, list):
        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        out = []
        for idx, choice in enumerate(choices):
            text = stringify_choice(choice)
            if not re.match(r"^[A-Z]\.\s", text) and len(choices) <= len(labels):
                text = f"{labels[idx]}. {text}"
            out.append(text)
        return out
    return []


def get_answer(row: dict[str, Any], choices: list[str]) -> str | None:
    answer = row.get("answer")
    if answer is None:
        answer = row.get("mc_answer_text")
    if answer is None and isinstance(row.get("mc_answer"), int):
        idx = int(row["mc_answer"])
        if 0 <= idx < len(choices):
            answer = choices[idx]
    if isinstance(answer, list):
        return "\n".join(str(x) for x in answer)
    return None if answer is None else str(answer)


def get_answer_label(row: dict[str, Any], answer: str | None) -> str | None:
    for key in ("answer_letter", "mc_answer_label"):
        if row.get(key) is not None:
            return str(row[key])
    if answer:
        match = re.match(r"^([A-Z])[\.\)]\s", answer.strip())
        if match:
            return match.group(1)
    if isinstance(row.get("mc_answer"), int):
        return chr(ord("A") + int(row["mc_answer"]))
    return None


def get_question(row: dict[str, Any]) -> str | None:
    for key in ("question", "query", "prompt", "instruction_text", "instructions", "instruction"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def get_prompt(row: dict[str, Any], question: str | None, choices: list[str]) -> str | None:
    for key in ("text_prompt", "prompt", "instructions", "instruction", "instruction_text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    if question and choices:
        return question + "\n\nChoices:\n" + "\n".join(choices)
    return question


def get_duration(row: dict[str, Any]) -> float | None:
    for key in ("duration_s", "duration_sec", "audio_duration_sec"):
        value = row.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def make_record(
    row: dict[str, Any],
    source: dict[str, Any],
    row_index: int,
    audio_paths: list[str],
    src_to_rel: dict[str, str],
) -> dict[str, Any]:
    choices = get_choices(row)
    answer = get_answer(row, choices)
    question = get_question(row)
    prompt = get_prompt(row, question, choices)
    source_id = (
        row.get("item_id")
        or row.get("bundle_item_id")
        or row.get("id")
        or row.get("index")
        or row.get("group_id")
        or row_index
    )
    benchmark = output_benchmark(source)
    task = source["benchmark"]
    subtask = row.get("subtask") or row.get("sub_task") or row.get("split") or row.get("_group")
    ability = get_ability(row, source)

    return {
        "id": f"{benchmark}__{source['subset']}__{slug(source_id)}",
        "benchmark": benchmark,
        "subset": source["subset"],
        "task": task,
        "subtask": None if subtask is None else str(subtask),
        "ability": ability,
        "source_id": str(source_id),
        "question": question,
        "prompt": prompt,
        "choices": choices,
        "answer": answer,
        "answer_label": get_answer_label(row, answer),
        "audio": audio_paths[0] if audio_paths else None,
        "audio_paths": audio_paths,
        "num_audio": len(audio_paths),
        "duration_s": get_duration(row),
        "source_manifest": Path(source["path"]).name,
        "metadata_json": json.dumps(
            sanitize_metadata(row, source, src_to_rel),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def croissant_field(field_id: str, data_type: str, description: str) -> dict[str, Any]:
    return {
        "@id": field_id,
        "@type": "cr:Field",
        "name": field_id,
        "description": description,
        "dataType": data_type,
        "source": {
            "fileObject": {"@id": "all_jsonl"},
            "extract": {"column": field_id},
        },
    }


def write_croissant(summary: dict[str, Any]) -> None:
    distributions: list[dict[str, Any]] = [
        {
            "@type": "cr:FileObject",
            "@id": "all_jsonl",
            "name": "all.jsonl",
            "description": "Unified benchmark manifest with one JSON object per row.",
            "contentUrl": f"{DATASET_RESOLVE_URL}/data/all.jsonl",
            "encodingFormat": "application/jsonlines",
            "sha256": sha256_file(DATA_DIR / "all.jsonl"),
        },
        {
            "@type": "cr:FileSet",
            "@id": "audio_files",
            "name": "audio",
            "description": "Audio files referenced by relative paths in the manifest.",
            "contentUrl": f"{DATASET_RESOLVE_URL}/audio/",
            "includes": "audio/**/*",
            "encodingFormat": ["audio/wav", "audio/mpeg", "audio/flac", "audio/ogg"],
        },
    ]
    for item in summary["subsets"]:
        distributions.append(
            {
                "@type": "cr:FileObject",
                "@id": f"{item['benchmark']}__{item['subset']}_jsonl",
                "name": item["filename"],
                "description": f"Rows for {item['benchmark']} / {item['subset']}.",
                "contentUrl": f"{DATASET_RESOLVE_URL}/data/{item['filename']}",
                "encodingFormat": "application/jsonlines",
                "sha256": sha256_file(DATA_DIR / item["filename"]),
            }
        )

    fields = [
        croissant_field("id", "sc:Text", "Stable unique identifier for the benchmark item."),
        croissant_field("benchmark", "sc:Text", "Top-level benchmark group."),
        croissant_field("subset", "sc:Text", "Source subset within a benchmark group."),
        croissant_field("task", "sc:Text", "Task name where available."),
        croissant_field("subtask", "sc:Text", "Subtask or source split where available."),
        croissant_field(
            "ability",
            "sc:Text",
            "Normalized CHC-style ability label from the benchmark taxonomy.",
        ),
        croissant_field("source_id", "sc:Text", "Identifier from the source manifest."),
        croissant_field("question", "sc:Text", "Question or query presented to the model."),
        croissant_field("prompt", "sc:Text", "Full prompt text, including instructions and choices when available."),
        croissant_field("choices", "sc:Text", "Multiple-choice options or allowed outputs."),
        croissant_field(
            "answer",
            "sc:Text",
            "Gold answer text. For multiple-choice tasks, a prediction is correct if it matches either this field or answer_label.",
        ),
        croissant_field(
            "answer_label",
            "sc:Text",
            "Gold answer option label for multiple-choice tasks where available. For MCQ scoring, matching answer or answer_label is counted as correct.",
        ),
        croissant_field("audio", "sc:Text", "Primary relative audio path."),
        croissant_field("audio_paths", "sc:Text", "All relative audio paths required by the item."),
        croissant_field("num_audio", "sc:Integer", "Number of audio files referenced by the item."),
        croissant_field("duration_s", "sc:Float", "Audio duration in seconds where available."),
        croissant_field("source_manifest", "sc:Text", "Original source manifest filename."),
        croissant_field("metadata_json", "sc:Text", "Sanitized original source row encoded as JSON text."),
    ]

    source_names = sorted(
        {
            f"{item['benchmark']}__{item['subset']}"
            for item in summary["subsets"]
        }
    )
    croissant = {
        "@context": {
            "@language": "en",
            "@vocab": "https://schema.org/",
            "citeAs": "cr:citeAs",
            "column": "cr:column",
            "conformsTo": "dct:conformsTo",
            "cr": "http://mlcommons.org/croissant/",
            "data": {
                "@id": "cr:data",
                "@type": "@json",
            },
            "dataType": {
                "@id": "cr:dataType",
                "@type": "@vocab",
            },
            "dct": "http://purl.org/dc/terms/",
            "extract": "cr:extract",
            "field": "cr:field",
            "fileObject": "cr:fileObject",
            "fileProperty": "cr:fileProperty",
            "fileSet": "cr:fileSet",
            "format": "cr:format",
            "includes": "cr:includes",
            "isLiveDataset": "cr:isLiveDataset",
            "jsonPath": "cr:jsonPath",
            "key": "cr:key",
            "md5": "cr:md5",
            "parentField": "cr:parentField",
            "path": "cr:path",
            "rai": "http://mlcommons.org/croissant/RAI/",
            "recordSet": "cr:recordSet",
            "references": "cr:references",
            "regex": "cr:regex",
            "repeated": "cr:repeated",
            "replace": "cr:replace",
            "equivalentProperty": "cr:equivalentProperty",
            "examples": "cr:examples",
            "samplingRate": "cr:samplingRate",
            "sc": "https://schema.org/",
            "separator": "cr:separator",
            "source": "cr:source",
            "subField": "cr:subField",
            "transform": "cr:transform",
            "prov": "http://www.w3.org/ns/prov#",
        },
        "@type": "sc:Dataset",
        "name": DATASET_NAME,
        "description": DATASET_DESCRIPTION,
        "url": DATASET_URL,
        "license": DATASET_LICENSE,
        "version": DATASET_VERSION,
        "datePublished": DATASET_DATE_PUBLISHED,
        "citeAs": DATASET_CITE_AS,
        "conformsTo": "http://mlcommons.org/croissant/1.1",
        "distribution": distributions,
        "recordSet": [
            {
                "@id": "rail_records",
                "@type": "cr:RecordSet",
                "name": "rail_records",
                "description": "Unified item-level records for the RAIL benchmark.",
                "field": fields,
            }
        ],
        "rai:dataLimitations": (
            "The benchmark is intended for evaluating audio and auditory-cognitive reasoning, "
            "not for training production systems or making decisions about individuals. "
            "Coverage varies across source datasets, tasks, languages, speakers, environments, "
            "and synthetic generation procedures; results may not generalize to all accents, "
            "recording conditions, cultures, clinical settings, or real-world auditory tasks. "
            "Some rows aggregate processed versions of external datasets and should be used "
            "within the license and consent constraints of those sources."
        ),
        "rai:dataBiases": (
            "Potential biases include source-dataset selection bias, English-centric prompts, "
            "uneven coverage of speakers, environments, tasks, audio qualities, and model-facing "
            "question styles. Synthetic or templated tasks may over-represent clean, well-formed "
            "instructions compared with naturally occurring audio interactions."
        ),
        "rai:personalSensitiveInformation": (
            "Some source audio may contain human speech and therefore may encode or reveal "
            "sensitive attributes such as language, accent, perceived gender, age, emotion, "
            "or cultural background. The benchmark should not be used to identify, profile, "
            "or make consequential inferences about individuals."
        ),
        "rai:dataUseCases": (
            "Validated use cases are benchmark evaluation and diagnostic analysis of audio "
            "reasoning systems across the included task categories. The dataset is not validated "
            "for model training, human-subject assessment, medical or accessibility diagnosis, "
            "surveillance, biometric identification, or high-stakes decision making."
        ),
        "rai:dataSocialImpact": (
            "Positive impact includes improving transparency and reproducibility for audio "
            "reasoning evaluation. Risks include overclaiming general audio understanding, "
            "amplifying source-dataset biases, or using speech attributes for profiling. "
            "Mitigations include preserving source metadata, documenting limitations, and "
            "recommending evaluation-only use."
        ),
        "rai:hasSyntheticData": True,
        "prov:wasDerivedFrom": source_names,
        "prov:wasGeneratedBy": [
            {
                "@type": "prov:Activity",
                "name": "RAIL benchmark normalization",
                "description": (
                    "Source manifests were normalized into a unified JSONL schema; audio files "
                    "were organized under relative paths; source rows were preserved in "
                    "metadata_json after sanitizing path fields."
                ),
            }
        ],
    }
    content = json.dumps(croissant, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    for filename in ("metadata.json", "croissant_rai_RAIL_Audio_Benchmark.json"):
        (RAIL_ROOT / filename).write_text(content, encoding="utf-8")


def write_readme(summary: dict[str, Any], skipped: list[dict[str, str]]) -> None:
    config_lines = [
        "---",
        "configs:",
        "- config_name: all",
        "  data_files:",
        "  - split: test",
        "    path: data/all.jsonl",
    ]
    for item in summary["subsets"]:
        config_lines.extend(
            [
                f"- config_name: {item['benchmark']}__{item['subset']}",
                "  data_files:",
                "  - split: test",
                f"    path: data/{item['filename']}",
            ]
        )
    config_lines.extend(["---", ""])

    body = [
        "# RAIL Audio Benchmark",
        "",
        "This folder is generated for direct Hugging Face dataset upload. "
        "Each row uses relative audio paths rooted at this repository folder.",
        "",
        "## NeurIPS / Croissant Metadata",
        "",
        "- `metadata.json` is a Croissant-style metadata file with core fields and minimal RAI fields.",
        "- `metadata.json` includes the Hugging Face dataset URL, CC-BY-4.0 license URL, checksums, and RAI fields.",
        "- `build_summary.json` contains build counts and skipped-source diagnostics.",
        "",
        "## Schema",
        "",
        "- `id`: stable row id.",
        "- `benchmark`, `subset`, `task`, `subtask`, `ability`: benchmark grouping fields.",
        "- `ability`: normalized CHC-style ability label from the benchmark taxonomy.",
        "- `question`, `prompt`, `choices`, `answer`, `answer_label`: evaluation fields.",
        "- For multiple-choice questions, predictions should be counted as correct if they match either `answer` or `answer_label`.",
        "- `audio`: first relative audio path for convenience.",
        "- `audio_paths`: all relative audio paths needed by the item.",
        "- `metadata_json`: sanitized source row as a JSON string; audio path fields are relative or null.",
        "",
        "## Counts",
        "",
        f"- Total rows: {summary['total_rows']}",
        f"- Unique audio files linked/copied: {summary['unique_audio_files']}",
        f"- Rows with no primary audio: {summary['rows_without_audio']}",
        "",
        "## Subsets",
        "",
    ]
    for item in summary["subsets"]:
        body.append(
            f"- `{item['benchmark']}__{item['subset']}`: "
            f"{item['rows']} rows, `{item['filename']}`"
        )
    if skipped:
        body.extend(["", "## Skipped Sources", ""])
        for item in skipped:
            body.append(f"- `{item['benchmark']}__{item['subset']}`: {item['reason']}")

    (RAIL_ROOT / "README.md").write_text("\n".join(config_lines + body) + "\n", encoding="utf-8")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    subsets = []
    skipped = []
    missing_audio: list[dict[str, Any]] = []
    copy_stats: Counter = Counter()
    used_relpaths: set[str] = set()
    src_to_rel: dict[str, str] = {}

    for source in SOURCES:
        source_path = Path(source["path"])
        try:
            rows = load_records(source)
        except PermissionError as exc:
            skipped.append(
                {
                    "benchmark": source["benchmark"],
                    "subset": source["subset"],
                    "path": str(source_path),
                    "reason": f"permission denied: {exc}",
                }
            )
            continue
        except FileNotFoundError as exc:
            skipped.append(
                {
                    "benchmark": source["benchmark"],
                    "subset": source["subset"],
                    "path": str(source_path),
                    "reason": f"not found: {exc}",
                }
            )
            continue

        out_rows = []
        try:
            for index, row in enumerate(rows):
                if not isinstance(row, dict):
                    continue
                audio_paths = collect_audio_paths(
                    row,
                    source,
                    used_relpaths,
                    src_to_rel,
                    copy_stats,
                    missing_audio,
                    primary_only=True,
                )
                # Copy path-like metadata audio as well so sanitized metadata does not point outside RAIL.
                collect_audio_paths(
                    row,
                    source,
                    used_relpaths,
                    src_to_rel,
                    copy_stats,
                    missing_audio,
                    primary_only=False,
                )
                out_rows.append(make_record(row, source, index, audio_paths, src_to_rel))
        except PermissionError as exc:
            skipped.append(
                {
                    "benchmark": source["benchmark"],
                    "subset": source["subset"],
                    "path": str(source_path),
                    "reason": f"audio permission denied: {exc}",
                }
            )
            continue

        benchmark = output_benchmark(source)
        filename = f"{benchmark}__{source['subset']}.jsonl"
        write_jsonl(DATA_DIR / filename, out_rows)
        subsets.append(
            {
                "benchmark": benchmark,
                "subset": source["subset"],
                "task": source["benchmark"],
                "rows": len(out_rows),
                "filename": filename,
            }
        )
        all_rows.extend(out_rows)

    write_jsonl(DATA_DIR / "all.jsonl", all_rows)

    summary = {
        "total_rows": len(all_rows),
        "unique_audio_files": len(src_to_rel),
        "rows_without_audio": sum(1 for row in all_rows if not row["audio_paths"]),
        "subsets": subsets,
        "copy_stats": dict(copy_stats),
        "skipped_sources": skipped,
        "missing_audio_count": len(missing_audio),
        "missing_audio_examples": missing_audio[:50],
    }
    (RAIL_ROOT / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_croissant(summary)
    write_readme(summary, skipped)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
