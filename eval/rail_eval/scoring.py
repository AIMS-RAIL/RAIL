"""Rule-based scoring (paper "ACC") and B-AUC.

The matching rules are ported from the scripts used for the paper so scores stay comparable:

* multiple_choice  `strict_match_mcq`   (eval/legacy/evaluation_metrices.py). The parsed answer
  may be a letter, a letter with content, or content only; content must contain the gold
  tokens in order and no token that only appears in other options.
* closed_label     `normalize_to_allowed` (paper efficiency/memory runner),
  then exact label match. Used for Processing Efficiency and the UM n-back items.
* open_ended       `strict_match_non_mcq` (numeric equality, number-in-text, or token equality).
* free_recall      strict token recall (eval/legacy/eval_memory_recall.py): multiset overlap of
  lowercase alphanumeric tokens of length >= 3 between response and gold items.
* B-AUC            normalised trapezoidal area of Acc<=b over reason-length budgets b = 0..50
  (paper Eq. 1). Reason length = number of [A-Za-z0-9]+ tokens in the parsed Reason segment;
  items without a recoverable reason stay in the denominator but never meet a budget.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BUDGET_GRID = list(range(0, 51))
LABEL_UNIT_RE = re.compile(r"[A-Za-z0-9]+")

# ---------------------------------------------------------------------------------------------
# Response parsing (paper reasoning runner, _extract_reason_and_answer)
# ---------------------------------------------------------------------------------------------

_ANSWER_LINE_RE = re.compile(r"^(?:final\s+)?answer\s*:", re.IGNORECASE)


def extract_reason_answer(raw_text: str) -> Dict[str, object]:
    txt = str(raw_text or "").strip()
    if not txt:
        return {"reason": "", "answer": "", "answer_field": False}
    m = re.search(r"Reason(?:ing)?\s*:\s*(.*?)\s*[;,]?\s*(?:Final\s+)?Answer\s*:\s*(.*)$", txt,
                  flags=re.IGNORECASE | re.DOTALL)
    if m:
        reason, ans_block = m.group(1).strip().rstrip(";,").strip(), m.group(2).strip()
    else:
        m2 = re.search(r"^(.*?)[;\n]\s*(?:Final\s+)?Answer\s*:\s*(.*)$", txt, flags=re.IGNORECASE | re.DOTALL)
        if m2:
            reason, ans_block = m2.group(1).strip().rstrip(";,").strip(), m2.group(2).strip()
        else:
            lines = [x.strip() for x in txt.splitlines() if x.strip()]
            reason, answer, found = "", "", False
            for line in lines:
                if _ANSWER_LINE_RE.match(line) and not answer:
                    answer = _ANSWER_LINE_RE.sub("", line).strip()
                    found = True
                elif "reason" in line.lower() and ":" in line and not reason:
                    reason = line.split(":", 1)[1].strip()
            if not answer and lines:
                answer = lines[-1]
            return {"reason": reason, "answer": answer, "answer_field": found}
    answer = next((line.strip() for line in ans_block.splitlines() if line.strip()), ans_block)
    return {"reason": reason, "answer": answer, "answer_field": True}


def reason_length(reason: str) -> Optional[int]:
    """Label-unit token count of the Reason segment; None when no reason was recovered."""
    reason = str(reason or "").strip()
    return len(LABEL_UNIT_RE.findall(reason)) if reason else None


# ---------------------------------------------------------------------------------------------
# Strict MCQ / open-ended matching (eval/legacy/evaluation_metrices.py)
# ---------------------------------------------------------------------------------------------

_STRIP_RE = re.compile(r"[^\w\s]", re.UNICODE)
_LETTER_ONLY_RE = re.compile(r"^\s*\(?([A-Za-z])\)?[.):]?\s*$")
_LETTER_CONTENT_RE = re.compile(r"^\s*\(?([A-Za-z])\)?[.):]\s*(\S.*)", re.DOTALL)


def _tokenize(text: str) -> List[str]:
    return [t for t in _STRIP_RE.sub(" ", str(text or "").lower()).split() if t]


def _extract_letter(text: str) -> Optional[str]:
    m = re.match(r"^\s*\(?([A-Za-z])\)?[.):]?\s*", str(text or "").strip())
    return m.group(1).upper() if m else None


def _extract_content(text: str) -> str:
    text = str(text or "").strip()
    m = _LETTER_CONTENT_RE.match(text)
    if m:
        return m.group(2).strip()
    return "" if _LETTER_ONLY_RE.match(text) else text


def _is_subsequence(needle: List[str], haystack: List[str]) -> bool:
    it = iter(haystack)
    return all(tok in it for tok in needle)


def _distractor_tokens(gt_letter: Optional[str], gt_content: str, choices: Sequence[str]) -> set:
    gt_set = set(_tokenize(gt_content))
    out = set()
    for choice in choices:
        c_letter = _extract_letter(choice)
        if gt_letter and c_letter and c_letter.upper() == gt_letter.upper():
            continue
        out.update(t for t in _tokenize(_extract_content(choice)) if t not in gt_set)
    return out


def strict_match_mcq(prediction: str, ground_truth: str, choices: Sequence[str]) -> bool:
    """`ground_truth` like "C. text"; `choices` like ["A. ...", "B. ..."]."""
    pred, gt = str(prediction or "").strip(), str(ground_truth or "").strip()
    if not pred or not gt:
        return False
    gt_letter, gt_content = _extract_letter(gt), _extract_content(gt)
    gt_tokens = _tokenize(gt_content)
    distractors = _distractor_tokens(gt_letter, gt_content, choices)
    if _LETTER_ONLY_RE.match(pred):
        pl = _extract_letter(pred)
        return pl is not None and gt_letter is not None and pl == gt_letter
    if _LETTER_CONTENT_RE.match(pred):
        pl = _extract_letter(pred)
        if not (pl is not None and gt_letter is not None and pl == gt_letter):
            return False
        if not gt_tokens:
            return True
        toks = _tokenize(_extract_content(pred))
        return _is_subsequence(gt_tokens, toks) and not any(t in distractors for t in toks)
    if not gt_tokens:
        pl = _extract_letter(pred)
        return pl is not None and gt_letter is not None and pl == gt_letter
    toks = _tokenize(pred)
    return _is_subsequence(gt_tokens, toks) and not any(t in distractors for t in toks)


def _to_number(text: str) -> Optional[float]:
    try:
        return float(str(text or "").strip().rstrip(".,;:!?"))
    except ValueError:
        return None


def _extract_numbers(text: str) -> List[float]:
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", str(text or ""))]


def strict_match_non_mcq(prediction: str, ground_truth: str) -> bool:
    gt_num = _to_number(ground_truth)
    if gt_num is not None:
        pred_num = _to_number(prediction)
        if pred_num is not None:
            return pred_num == gt_num
        return gt_num in _extract_numbers(prediction)
    pred_toks, gt_toks = _tokenize(prediction), _tokenize(ground_truth)
    return bool(gt_toks) and pred_toks == gt_toks


# ---------------------------------------------------------------------------------------------
# Closed-label normalisation (paper efficiency/memory runner)
# ---------------------------------------------------------------------------------------------


def normalize_to_allowed(raw_answer: str, allowed: Sequence[str]) -> str:
    txt = str(raw_answer or "").strip()
    if not txt or not allowed:
        return txt
    low = {x.lower(): x for x in allowed}
    if txt.lower() in low:
        return low[txt.lower()]
    cand = sorted({low[t.lower()] for t in re.findall(r"[A-Za-z0-9_=;]+", txt) if t.lower() in low})
    if len(cand) == 1:
        return cand[0]
    hits = sorted({a for a in allowed if a.lower() in txt.lower()})
    return hits[0] if len(hits) == 1 else ""


# ---------------------------------------------------------------------------------------------
# Free recall (eval/legacy/eval_memory_recall.py)
# ---------------------------------------------------------------------------------------------

_RECALL_TOKEN_RE = re.compile(r"[a-z0-9]+")
_ITEM_SPLIT_RE = re.compile(r"(?:^|\n)\s*(?:[-*]\s+|\d+\.\s+)?")


def _recall_tokens(text: str) -> List[str]:
    return [t for t in _RECALL_TOKEN_RE.findall(str(text or "").lower()) if len(t) >= 3]


def split_items(text: str) -> List[str]:
    return [p.strip() for p in _ITEM_SPLIT_RE.split(str(text or "")) if p and p.strip()]


def recall_counts(response: str, gold: str) -> Tuple[int, int]:
    pred: Dict[str, int] = {}
    for item in split_items(response):
        for t in _recall_tokens(item):
            pred[t] = pred.get(t, 0) + 1
    gold_tokens = [t for item in split_items(gold) for t in _recall_tokens(item)]
    counts: Dict[str, int] = {}
    for t in gold_tokens:
        counts[t] = counts.get(t, 0) + 1
    hit = sum(min(c, pred.get(t, 0)) for t, c in counts.items())
    return hit, len(gold_tokens)


# ---------------------------------------------------------------------------------------------
# Per-item scoring
# ---------------------------------------------------------------------------------------------


def gold_reference(item: dict) -> str:
    """Gold answer as shown to the LLM judge and in reports."""
    if item["answer_format"] == "multiple_choice" and item.get("answer_label"):
        return f"{item['answer_label']}. {item['answer']}"
    return str(item["answer"])


def score_item(item: dict, response: str) -> Dict[str, object]:
    parsed = extract_reason_answer(response)
    answer = str(parsed["answer"])
    fmt = item["answer_format"]
    out: Dict[str, object] = {
        "reason": parsed["reason"],
        "answer_parsed": answer,
        "answer_field": parsed["answer_field"],
        "reason_len": reason_length(str(parsed["reason"])),
    }
    if fmt == "multiple_choice":
        choices = [f"{LETTERS[i]}. {c}" for i, c in enumerate(item["choices"])]
        out["correct"] = float(bool(item.get("answer_label")) and
                               strict_match_mcq(answer, gold_reference(item), choices))
    elif fmt == "closed_label":
        label = normalize_to_allowed(answer, item["choices"])
        if not label and not parsed["answer_field"]:
            label = normalize_to_allowed(response, item["choices"])
        out["answer_normalized"] = label
        out["correct"] = float(label == item["answer"])
    elif fmt == "open_ended":
        out["correct"] = float(strict_match_non_mcq(answer, item["answer"]))
    elif fmt == "free_recall":
        hit, total = recall_counts(response, item["answer"])
        out.update(recall_hit=hit, recall_total=total)
        out["correct"] = hit / total if total else 0.0
    else:
        raise ValueError(f"unknown answer_format {fmt!r}")
    return out


# ---------------------------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------------------------


def budget_curve(correct: Sequence[float], lengths: Sequence[Optional[int]],
                 grid: Sequence[int] = BUDGET_GRID) -> List[float]:
    n = len(correct)
    if not n:
        return [0.0 for _ in grid]
    return [sum(c for c, l in zip(correct, lengths) if l is not None and l <= b) / n for b in grid]


def bauc(correct: Sequence[float], lengths: Sequence[Optional[int]],
         grid: Sequence[int] = BUDGET_GRID) -> float:
    curve = budget_curve(correct, lengths, grid)
    span = grid[-1] - grid[0]
    return sum((curve[i] + curve[i + 1]) / 2 for i in range(len(curve) - 1)) / span if span else 0.0


def mean(values: Sequence[float]) -> Optional[float]:
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(values) / len(values) if values else None


def median(values: Sequence[float]) -> Optional[float]:
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    mid = len(values) // 2
    return float(values[mid]) if len(values) % 2 else (values[mid - 1] + values[mid]) / 2
