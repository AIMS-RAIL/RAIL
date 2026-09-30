"""
evaluation_metrices.py
======================
Two-part evaluation suite for AudioCog benchmark result files.

Input format
------------
Each result file is either:
  • a JSON file  (.json)  — containing a single JSON array of record objects, or
  • a JSONL file (.jsonl) — one JSON object per line.

Each record must contain at minimum:
  "answer"       — ground-truth answer (string).
                   For MCQ items this is typically "B. happy"; for non-MCQ it is
                   a plain string or number such as "1.5".
  "answer_text"  — model's extracted answer used by strict match.
  "model_output" — model's full raw output used by LLM-as-judge.
  "choices"      — (optional) list of MCQ option strings, e.g.
                   ["A. happy", "B. sad", "C. angry"].
                   Omit or set to [] / null for non-MCQ items.
  "ok"           — (optional) boolean; records where ok=false are skipped.

Part 1 — Strict Match
    MCQ items (choices present):
      A) prediction is a single letter  →  compare letter only
      B) prediction is letter + content →  letter must match AND content
                                            must pass token-subsequence check
      C) prediction is content only     →  content must pass token-subsequence
                                            check (no letter comparison)
    Token-subsequence check:
      - All ground-truth content tokens must appear in the prediction in order.
      - The prediction must contain no "distractor tokens" (tokens from other
        choices that are absent from the ground-truth content).
    Non-MCQ items:
      1. If the ground truth is a bare number, compare numerically
         ("1.5" == "1.50" == "1.5."; "10 liters" matches GT "10").
      2. Otherwise, strip all special characters and compare token lists
         ("same-speaker" == "same speaker").

Part 2 — LLM as Judge
    Uses GPT-5.4 (or any OpenAI-compatible model) to make a binary
    correct / incorrect / unresolved judgment on the model's raw output
    (model_output field) against the gold answer.
    The judge is instructed to identify the final committed answer from the
    model response (ignoring intermediate reasoning) and return "true" if it
    matches the gold answer semantically, "false" otherwise.
    Requires the OPENAI_API_KEY environment variable (or an injected client).

CLI usage
---------
    python evaluation_metrices.py results.json
    python evaluation_metrices.py results.jsonl
    python evaluation_metrices.py results_dir/              # all .json/.jsonl files
    python evaluation_metrices.py results.json --group task_type
    python evaluation_metrices.py results.json --mode llm   # LLM-as-judge
    python evaluation_metrices.py results.json --mode both  # strict + LLM
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Tokenisation helpers
# ─────────────────────────────────────────────────────────────────────────────

# Remove every character that is not a Unicode word character (letters, digits,
# underscore) or ASCII whitespace.  This covers ".", "-", ",", "⟨", "⟩", etc.
_STRIP_RE = re.compile(r"[^\w\s]", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """
    Lowercase, strip all special characters (punctuation, brackets, hyphens,
    dots …), and return the remaining non-empty word tokens.

    Examples
    --------
    "A. good"          → ["a", "good"]
    "⟨Pear, Pear⟩"    → ["pear", "pear"]
    "same-speaker"     → ["same", "speaker"]

    Note: numeric strings like "1.5" are handled by _to_number / strict_match_non_mcq
    before tokenisation is reached, so the ["1","5"] split never affects them.
    """
    cleaned = _STRIP_RE.sub(" ", str(text or "").lower())
    return [t for t in cleaned.split() if t]


# ─────────────────────────────────────────────────────────────────────────────
# Option-letter helpers
# ─────────────────────────────────────────────────────────────────────────────

# Matches a lone letter: "A", "A.", "(A)", "A)", "a."
_LETTER_ONLY_RE = re.compile(r"^\s*\(?([A-Za-z])\)?[.):]?\s*$")

# Matches letter followed by content: "A. good", "A) good", "A good"
_LETTER_CONTENT_RE = re.compile(r"^\s*\(?([A-Za-z])\)?[.):]\s*(\S.*)", re.DOTALL)


def _extract_letter(text: str) -> Optional[str]:
    """Return upper-cased option letter if text starts with one, else None."""
    text = str(text or "").strip()
    m = re.match(r"^\s*\(?([A-Za-z])\)?[.):]?\s*", text)
    if m:
        return m.group(1).upper()
    return None


def _extract_content(text: str) -> str:
    """Strip a leading option letter (if present) and return the content."""
    text = str(text or "").strip()
    m = _LETTER_CONTENT_RE.match(text)
    if m:
        return m.group(2).strip()
    m2 = _LETTER_ONLY_RE.match(text)
    if m2:
        return ""
    return text


def _is_letter_only(text: str) -> bool:
    return bool(_LETTER_ONLY_RE.match(str(text or "").strip()))


def _is_letter_with_content(text: str) -> bool:
    return bool(_LETTER_CONTENT_RE.match(str(text or "").strip()))


# ─────────────────────────────────────────────────────────────────────────────
# Subsequence check
# ─────────────────────────────────────────────────────────────────────────────

def _is_subsequence(needle: list[str], haystack: list[str]) -> bool:
    """Return True iff every element of needle appears in haystack in order."""
    it = iter(haystack)
    return all(tok in it for tok in needle)


# ─────────────────────────────────────────────────────────────────────────────
# Distractor-token set builder
# ─────────────────────────────────────────────────────────────────────────────

def _distractor_tokens(gt_letter: Optional[str], gt_content: str,
                        choices: list[str]) -> set[str]:
    """
    Collect tokens that appear in competing choices but NOT in the ground-truth
    content.  These tokens should NOT appear in a valid prediction.
    """
    gt_token_set = set(_tokenize(gt_content))
    distractors: set[str] = set()
    for choice in choices:
        c_letter = _extract_letter(choice)
        # Skip the ground-truth choice itself
        if gt_letter and c_letter and c_letter.upper() == gt_letter.upper():
            continue
        c_content = _extract_content(choice)
        for tok in _tokenize(c_content):
            if tok not in gt_token_set:
                distractors.add(tok)
    return distractors


# ─────────────────────────────────────────────────────────────────────────────
# Core strict-match logic
# ─────────────────────────────────────────────────────────────────────────────

def strict_match_mcq(prediction: str, ground_truth: str,
                      choices: list[str]) -> bool:
    """
    Strict match for multiple-choice items.

    Parameters
    ----------
    prediction  : model's extracted prediction string
    ground_truth : correct answer, e.g. "C. ⟨Pear, Pear, …⟩" or "C"
    choices      : list of all option strings, e.g. ["A. …", "B. …", …]
    """
    pred = str(prediction or "").strip()
    gt   = str(ground_truth  or "").strip()

    if not pred or not gt:
        return False

    gt_letter  = _extract_letter(gt)
    gt_content = _extract_content(gt)
    gt_tokens  = _tokenize(gt_content)

    distractors = _distractor_tokens(gt_letter, gt_content, choices)

    # ── Case A: prediction is a lone letter ──────────────────────────────────
    if _is_letter_only(pred):
        pred_letter = _extract_letter(pred)
        return (pred_letter is not None and gt_letter is not None
                and pred_letter.upper() == gt_letter.upper())

    # ── Case B: prediction is letter + content ───────────────────────────────
    if _is_letter_with_content(pred):
        pred_letter  = _extract_letter(pred)
        letter_ok    = (pred_letter is not None and gt_letter is not None
                        and pred_letter.upper() == gt_letter.upper())
        if not letter_ok:
            return False
        if not gt_tokens:          # ground truth has no content tokens → letter match suffices
            return True
        pred_content_tokens = _tokenize(_extract_content(pred))
        return (_is_subsequence(gt_tokens, pred_content_tokens)
                and not any(t in distractors for t in pred_content_tokens))

    # ── Case C: prediction is content only (no letter prefix) ────────────────
    if not gt_tokens:
        # GT has no content part; fall back to letter comparison on prediction
        pred_letter = _extract_letter(pred)
        return (pred_letter is not None and gt_letter is not None
                and pred_letter.upper() == gt_letter.upper())

    pred_tokens = _tokenize(pred)
    return (_is_subsequence(gt_tokens, pred_tokens)
            and not any(t in distractors for t in pred_tokens))


def _to_number(text: str) -> Optional[float]:
    """
    Try to parse the entire text as a single number.
    Strips surrounding whitespace and trailing punctuation first.
    Returns None if the text contains anything other than a number.
    """
    s = str(text or "").strip().rstrip(".,;:!?")
    try:
        return float(s)
    except ValueError:
        return None


def _extract_numbers(text: str) -> list[float]:
    """
    Extract all numbers (integers or decimals, optionally negative) that
    appear anywhere in text.  Used when the ground truth is a pure number
    but the model wraps it in words (e.g. "10 liters left").
    """
    nums = []
    for m in re.findall(r"-?\d+(?:\.\d+)?", str(text or "")):
        try:
            nums.append(float(m))
        except ValueError:
            pass
    return nums


def strict_match_non_mcq(prediction: str, ground_truth: str) -> bool:
    """
    Match for open (non-MCQ) answers, with three strategies applied in order:

    1. Pure-numeric comparison — if the ground truth is a bare number and the
       prediction also parses as a bare number, compare as floats.
       ("1.5" == "1.50" == "1.5.")

    2. Number-in-text — if the ground truth is a bare number but the prediction
       contains additional words (e.g. "10 liters left"), accept if the exact
       numeric value appears anywhere inside the prediction.
       ("10" matches "10 liters left" but not "100 liters")

    3. Token-level comparison — for non-numeric ground truths, lowercase and
       strip all special characters before comparing token lists, so
       "same-speaker" == "same speaker" and "⟨yes⟩" == "yes".
    """
    gt_num = _to_number(ground_truth)

    if gt_num is not None:
        # Strategy 1: prediction is also a bare number
        pred_num = _to_number(prediction)
        if pred_num is not None:
            return pred_num == gt_num
        # Strategy 2: GT is a number but prediction has extra words
        return gt_num in _extract_numbers(prediction)

    # Strategy 3: non-numeric ground truth → token comparison
    pred_toks = _tokenize(prediction)
    gt_toks   = _tokenize(ground_truth)
    return bool(gt_toks) and pred_toks == gt_toks


def strict_match(item: dict) -> bool:
    """
    Compute strict match for one result record.

    Expects fields: prediction, answer (ground truth), choices (optional).
    """
    prediction  = str(item.get("answer_text")  or "").strip()
    ground_truth = str(item.get("answer") or item.get("ground_truth") or "").strip()
    choices      = item.get("choices") or []

    if choices:
        return strict_match_mcq(prediction, ground_truth, choices)
    return strict_match_non_mcq(prediction, ground_truth)


# ─────────────────────────────────────────────────────────────────────────────
# LLM-as-Judge
# ─────────────────────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = """\
You are a strict and impartial evaluator for benchmark outputs.

Instructions:
1. Identify the final answer expressed in the model response.
2. The final answer may be expressed directly, indirectly, or as a paraphrase.
3. Compare the model's final answer with the gold answer for semantic equivalence.
4. Do not grade reasoning quality. Only judge whether the final answer matches \
the gold answer.

Respond with EXACTLY one word — nothing else:
  true   — the model's final answer matches the gold answer
  false  — the model's final answer does not match, or no definite answer was given\
"""

_JUDGE_USER_TMPL = (
    "Gold answer:      {reference}\n"
    "Model response:   {prediction}\n\n"
    "Does the model's final answer match the gold answer?\n"
    "Reply with exactly one word: true or false."
)

_VERDICT_RE = re.compile(r"\b(true|false)\b", re.IGNORECASE)


def llm_judge(
    item: dict,
    *,
    client=None,
    model: str = "gpt-5.4",
) -> dict:
    """
    LLM-as-Judge evaluation for one result record.

    The judge reads the model's full raw output (``model_output``) and makes a
    binary correct / incorrect / unresolved verdict against the gold answer.

    Parameters
    ----------
    item   : result record; uses ``model_output`` (full model response) and
             ``answer`` / ``ground_truth`` (gold reference).
    client : an ``openai.OpenAI`` instance.  If None, one is created
             automatically from the ``OPENAI_API_KEY`` environment variable.
    model  : judge model (default ``gpt-5.4``).

    Returns
    -------
    dict with keys:
        correct (bool)    — True if the judge replied "true"
        raw     (str)     — raw text returned by the judge
        error   (str|None)— error message if the API call failed
    """
    import os
    try:
        import openai as _openai
    except ImportError as exc:
        return {"correct": False, "raw": "", "error": f"openai not installed: {exc}"}

    prediction = str(item.get("model_output") or "").strip()
    reference  = str(item.get("answer") or item.get("ground_truth") or "").strip()

    if not prediction or not reference:
        return {"correct": False, "raw": "",
                "error": "empty model_output or reference"}

    if client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return {"correct": False, "raw": "",
                    "error": "OPENAI_API_KEY not set and no client provided"}
        client = _openai.OpenAI(api_key=api_key)

    user_msg = _JUDGE_USER_TMPL.format(reference=reference, prediction=prediction)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0,
            max_completion_tokens=8,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as exc:
        return {"correct": False, "raw": "", "error": str(exc)}

    m = _VERDICT_RE.search(raw)
    correct = m.group(1).lower() == "true" if m else False
    return {"correct": correct, "raw": raw, "error": None}


# ─────────────────────────────────────────────────────────────────────────────
# File loading (JSON or JSONL)
# ─────────────────────────────────────────────────────────────────────────────

def load_records(path: pathlib.Path) -> list[dict]:
    """
    Load result records from a JSON or JSONL file.

    JSON  (.json)  — expects a top-level array; a bare object is also accepted
                     and wrapped in a list.
    JSONL (.jsonl) — one JSON object per non-empty line; blank lines ignored.
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        records = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records
    # Default: JSON
    data = json.loads(text)
    return data if isinstance(data, list) else [data]


# ─────────────────────────────────────────────────────────────────────────────
# File-level evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _save_to_judge_results(
    records: list[dict],
    path: pathlib.Path,
    judge_results_dir: pathlib.Path,
    mode: str,
) -> None:
    """
    Merge evaluated records into judge_results/{capability}/{model}.json.

    Records are matched by position.  Only the fields written by the current
    mode are touched; any other existing fields are left unchanged.
      strict → updates pattern_match_correct
      llm    → updates LLM-as-judge_correct
      both   → updates both
    """
    # Group source records by cognitive_capability
    cap_groups: dict[str, list] = defaultdict(list)
    for rec in records:
        cap = str(rec.get("cognitive_capability", "unknown"))
        cap_groups[cap].append(rec)

    do_strict = mode in ("strict", "both")
    do_llm    = mode in ("llm",    "both")

    for cap, src_recs in cap_groups.items():
        cap_dir  = judge_results_dir / cap
        cap_dir.mkdir(parents=True, exist_ok=True)
        out_file = cap_dir / path.name

        if out_file.exists():
            # File already exists — merge: overwrite only the current-mode fields
            existing = json.loads(out_file.read_text(encoding="utf-8"))
            for i, src in enumerate(src_recs):
                if i >= len(existing):
                    existing.append(src)
                    continue
                if do_strict and "pattern_match_correct" in src:
                    existing[i]["pattern_match_correct"] = src["pattern_match_correct"]
                if do_llm and "LLM-as-judge_correct" in src:
                    existing[i]["LLM-as-judge_correct"] = src["LLM-as-judge_correct"]
            out_recs = existing
        else:
            out_recs = src_recs

        out_file.write_text(
            json.dumps(out_recs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def update_scores_json(
    judge_results_dir: pathlib.Path,
    scores_json: pathlib.Path,
) -> None:
    """
    Recompute per-model accuracy from all files in judge_results/ and write
    scores.json.

    For each capability sub-directory the function reads every model file,
    counts correct records for both strict (pattern_match_correct) and LLM
    (LLM-as-judge_correct), and builds / updates the scores table.

    Plain columns  (e.g. ``induction``)       ← pattern_match_correct
    ``_llm`` columns (e.g. ``induction_llm``) ← LLM-as-judge_correct
    ``overall`` / ``overall_llm``             ← mean across all capabilities
    """
    CAPABILITIES = ["quantitative_reasoning", "induction", "sequential_reasoning"]

    # Load existing scores so we preserve rows for models we don't touch
    existing: dict[str, dict] = {}
    if scores_json.exists():
        raw = json.loads(scores_json.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            for row in raw:
                if isinstance(row, dict) and "model" in row:
                    existing[row["model"]] = row
        elif isinstance(raw, dict):
            existing.update(raw)

    for cap in CAPABILITIES:
        cap_dir = judge_results_dir / cap
        if not cap_dir.is_dir():
            continue
        for f in sorted(list(cap_dir.glob("*.json")) + list(cap_dir.glob("*.jsonl"))):
            model_name = f.stem
            records = json.loads(f.read_text(encoding="utf-8"))
            active = [r for r in records if r.get("ok", True)]
            n = len(active)
            if n == 0:
                continue

            row = existing.setdefault(model_name, {"model": model_name})

            # Strict / pattern match
            if any("pattern_match_correct" in r for r in active):
                correct = sum(1 for r in active if r.get("pattern_match_correct"))
                row[cap] = round(correct / n, 4)

            # LLM-as-judge
            if any("LLM-as-judge_correct" in r for r in active):
                correct = sum(1 for r in active if r.get("LLM-as-judge_correct"))
                row[f"{cap}_llm"] = round(correct / n, 4)

    # Recompute overall columns
    for row in existing.values():
        strict_vals = [row[c] for c in CAPABILITIES if c in row]
        llm_vals    = [row[f"{c}_llm"] for c in CAPABILITIES if f"{c}_llm" in row]
        if strict_vals:
            row["overall"] = round(sum(strict_vals) / len(strict_vals), 4)
        if llm_vals:
            row["overall_llm"] = round(sum(llm_vals) / len(llm_vals), 4)

    scores_json.write_text(
        json.dumps(list(existing.values()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[scores] Updated {scores_json}  ({len(existing)} models)")


def evaluate_file(
    path: pathlib.Path,
    group_by: str = "cognitive_capability",
    mode: str = "strict",          # "strict" | "llm" | "both"
    llm_client=None,
    llm_model: str = "gpt-5.4",
    judge_results_dir: Optional[pathlib.Path] = None,
    scores_json: Optional[pathlib.Path] = None,
) -> dict:
    """
    Load a result JSON file, compute accuracy, and optionally persist results.

    Parameters
    ----------
    path              : path to the JSON result file
    group_by          : item field to break down per-category stats
    mode              : "strict" — strict match only (always rewrites
                                   pattern_match_correct)
                        "llm"    — LLM-as-judge only (always rewrites
                                   LLM-as-judge_correct)
                        "both"   — run both
    llm_client        : openai.OpenAI instance (auto-created from env var if None)
    llm_model         : judge model name
    judge_results_dir : if set, merge results into
                        judge_results_dir/{capability}/{model}.json —
                        existing fields for the OTHER mode are preserved
    scores_json       : if set, recompute and overwrite scores.json after saving

    Returns
    -------
    dict with keys: file, total,
        strict_correct, strict_accuracy (when mode in strict/both),
        llm_correct, llm_accuracy (when mode in llm/both),
        per_group: {group_value: {total, strict_correct?, llm_correct?, …}}
    """
    records = load_records(path)

    do_strict = mode in ("strict", "both")
    do_llm    = mode in ("llm",    "both")

    total = 0
    strict_correct = 0
    llm_correct    = 0

    per_group: dict[str, dict] = defaultdict(lambda: {
        "total": 0,
        "strict_correct": 0,
        "llm_correct": 0,
    })

    # ── Strict match (fast — always rewrites pattern_match_correct) ──────────
    active_records = []
    for item in records:
        if not item.get("ok", True):
            continue
        total += 1
        active_records.append(item)
        group_val = str(item.get(group_by, "unknown"))
        pg = per_group[group_val]
        pg["total"] += 1

        if do_strict:
            sm = strict_match(item)
            item["pattern_match_correct"] = bool(sm)
            strict_correct      += sm
            pg["strict_correct"] += sm

    # ── LLM-as-judge (I/O-bound — always rewrites LLM-as-judge_correct) ──────
    if do_llm and active_records:
        n_workers = min(32, len(active_records))

        def _judge_item(item):
            return item, llm_judge(item, client=llm_client, model=llm_model)

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_judge_item, item): item
                       for item in active_records}
            done = 0
            for fut in as_completed(futures):
                item, jresult = fut.result()
                passed = bool(jresult["correct"])
                item["LLM-as-judge_correct"] = passed
                group_val = str(item.get(group_by, "unknown"))
                llm_correct          += passed
                per_group[group_val]["llm_correct"] += passed
                if jresult["error"]:
                    print(f"[WARN] LLM judge error for item "
                          f"{item.get('item_id','?')}: {jresult['error']}",
                          file=sys.stderr)
                done += 1
                if done % 50 == 0 or done == len(active_records):
                    print(f"  [LLM judge] {done}/{len(active_records)} done "
                          f"({llm_correct} correct so far)", flush=True)

    # ── Persist results ───────────────────────────────────────────────────────
    if judge_results_dir is not None:
        _save_to_judge_results(records, path, judge_results_dir, mode)

    out: dict = {"file": path.name, "total": total}

    if do_strict:
        out["strict_correct"]  = strict_correct
        out["strict_accuracy"] = strict_correct / total if total else 0.0

    if do_llm:
        out["llm_correct"]  = llm_correct
        out["llm_accuracy"] = llm_correct / total if total else 0.0

    # Finalise per-group stats
    for g in per_group.values():
        n = g["total"]
        if do_strict:
            g["strict_accuracy"] = g["strict_correct"] / n if n else 0.0
        if do_llm:
            g["llm_accuracy"] = g["llm_correct"] / n if n else 0.0

    out["per_group"] = dict(per_group)
    return out


def evaluate_directory(
    directory: pathlib.Path,
    group_by: str = "cognitive_capability",
    mode: str = "strict",
    llm_client=None,
    llm_model: str = "gpt-5.4",
    judge_results_dir: Optional[pathlib.Path] = None,
    scores_json: Optional[pathlib.Path] = None,
) -> list[dict]:
    """Evaluate every *.json file in a directory."""
    results = []
    paths = sorted(directory.glob("*.json")) + sorted(directory.glob("*.jsonl"))
    for p in sorted(paths):
        try:
            results.append(evaluate_file(
                p, group_by=group_by, mode=mode,
                llm_client=llm_client, llm_model=llm_model,
                judge_results_dir=judge_results_dir,
                scores_json=None,   # update scores once after all files
            ))
        except Exception as e:
            print(f"[WARN] Could not evaluate {p.name}: {e}", file=sys.stderr)

    # Update scores.json once after the whole directory is done
    if judge_results_dir is not None and scores_json is not None:
        update_scores_json(judge_results_dir, scores_json)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _print_result(r: dict) -> None:
    do_strict = "strict_accuracy" in r
    do_llm    = "llm_accuracy"    in r

    print(f"\n{'─'*60}")
    print(f"File : {r['file']}")

    if do_strict:
        print(f"Strict : {r['strict_correct']}/{r['total']}"
              f"  acc={r['strict_accuracy']:.3f}")
    if do_llm:
        print(f"LLM    : {r['llm_correct']}/{r['total']}"
              f"  acc={r['llm_accuracy']:.3f}")

    if r.get("per_group"):
        print("By group:")
        for grp, s in sorted(r["per_group"].items()):
            line = f"  {grp:<45}  n={s['total']}"
            if do_strict:
                line += (f"  strict={s['strict_correct']}/{s['total']}"
                         f" ({s['strict_accuracy']:.3f})")
            if do_llm:
                line += (f"  llm={s['llm_correct']}/{s['total']}"
                         f" ({s['llm_accuracy']:.3f})")
            print(line)


def main() -> None:
    import os

    _EVAL_DIR = pathlib.Path(__file__).parent
    _DEFAULT_JUDGE_DIR  = _EVAL_DIR / "judge_results"
    _DEFAULT_SCORES_JSON = _DEFAULT_JUDGE_DIR / "scores.json"

    parser = argparse.ArgumentParser(description="AudioCog evaluation metrics.")
    parser.add_argument("path", help="Result JSON file or directory of JSON files.")
    parser.add_argument("--group", default="cognitive_capability",
                        help="Field to group per-category accuracy by "
                             "(default: cognitive_capability).")
    parser.add_argument("--mode", default="strict",
                        choices=["strict", "llm", "both"],
                        help="Evaluation mode: strict match, LLM-as-judge, or both "
                             "(default: strict).")
    parser.add_argument("--llm-model", default="gpt-5.4",
                        help="Judge model for LLM-as-judge mode (default: gpt-5.4).")
    parser.add_argument(
        "--judge-results-dir",
        default=str(_DEFAULT_JUDGE_DIR),
        help="Root directory where per-capability result files are stored. "
             "Results are merged into {dir}/{capability}/{model}.json. "
             f"(default: {_DEFAULT_JUDGE_DIR})",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write results back to judge_results/ or update scores.json.",
    )
    parser.add_argument(
        "--scores-json",
        default=str(_DEFAULT_SCORES_JSON),
        help=f"Path to scores.json to update after saving. "
             f"(default: {_DEFAULT_SCORES_JSON})",
    )
    args = parser.parse_args()

    judge_results_dir = None if args.no_save else pathlib.Path(args.judge_results_dir)
    scores_json       = None if args.no_save else pathlib.Path(args.scores_json)

    # Build LLM client only when needed
    llm_client = None
    if args.mode in ("llm", "both"):
        try:
            import openai as _openai
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                print("[ERROR] OPENAI_API_KEY is not set.", file=sys.stderr)
                sys.exit(1)
            llm_client = _openai.OpenAI(api_key=api_key)
        except ImportError:
            print("[ERROR] openai package not installed. "
                  "Run: pip install openai", file=sys.stderr)
            sys.exit(1)

    kwargs = dict(
        group_by=args.group, mode=args.mode,
        llm_client=llm_client, llm_model=args.llm_model,
        judge_results_dir=judge_results_dir,
        scores_json=scores_json,
    )

    p = pathlib.Path(args.path)
    if p.is_dir():
        results = evaluate_directory(p, **kwargs)
        for r in results:
            _print_result(r)
        if results:
            tot = sum(r["total"] for r in results)
            print(f"\n{'='*60}")
            print(f"TOTAL across {len(results)} files  n={tot}")
            if "strict_accuracy" in results[0]:
                cor = sum(r["strict_correct"] for r in results)
                print(f"  Strict : {cor}/{tot}  acc={cor/tot:.3f}" if tot else "")
            if "llm_accuracy" in results[0]:
                cor = sum(r["llm_correct"] for r in results)
                print(f"  LLM    : {cor}/{tot}  acc={cor/tot:.3f}" if tot else "")
    elif p.is_file():
        r = evaluate_file(p, **kwargs)
        # Single-file: update scores.json manually after save
        if judge_results_dir is not None and scores_json is not None:
            update_scores_json(judge_results_dir, scores_json)
        _print_result(r)
    else:
        print(f"[ERROR] Path not found: {p}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
