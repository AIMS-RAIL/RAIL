"""Aggregate scored items and render the English evaluation report."""

from __future__ import annotations

import datetime as _dt
from collections import OrderedDict
from typing import Dict, List, Optional

from .scoring import BUDGET_GRID, bauc, budget_curve, mean, median

CAPABILITIES = OrderedDict([
    ("Ga", "Auditory Processing"),
    ("Gf", "Reasoning"),
    ("Gsm/Glr", "Memory"),
    ("Gs", "Processing Efficiency"),
    ("Gc/Gkn", "Knowledge"),
])
FULL_TASK_COUNTS = {"Ga": 1170, "Gf": 322, "Gsm/Glr": 1000, "Gs": 1800, "Gc/Gkn": 1014}
CHECKPOINTS = (5, 10, 20, 50)
# How the paper's main table combines tasks into one capability score (checked against the
# published Qwen2-Audio numbers): item-weighted for Ga/Gf/Gc, mean over tasks for Memory and Gs.
HEADLINE_POOLING = {"Ga": "items", "Gf": "items", "Gsm/Glr": "tasks", "Gs": "tasks", "Gc/Gkn": "items"}


def _pct(x: Optional[float], digits: int = 2) -> str:
    return "—" if x is None else f"{100 * x:.{digits}f}"


def _num(x: Optional[float], digits: int = 1) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def _group_stats(rows: List[dict], efficiency: bool) -> Dict[str, object]:
    correct = [float(r["correct"]) for r in rows]
    lengths = [r.get("reason_len") for r in rows]
    judged = [r for r in rows if r.get("judge_correct") is not None]
    stats: Dict[str, object] = {
        "n": len(rows),
        "acc": mean(correct),
        "judge_acc": mean([float(r["judge_correct"]) for r in judged]) if judged else None,
        "answer_field_rate": mean([float(bool(r.get("answer_field"))) for r in rows]),
        "reason_rate": mean([float(r.get("reason_len") is not None) for r in rows]),
        "mean_reason_len": mean([float(x) for x in lengths if x is not None]),
        "median_reason_len": median([x for x in lengths if x is not None]),
        "errors": sum(1 for r in rows if r.get("error")),
        "hit_max_new_tokens": sum(1 for r in rows if r.get("hit_max_new_tokens")),
        "truncated_items": sum(1 for r in rows if (r.get("n_clips_truncated") or 0) > 0),
    }
    if rows and all(r["answer_format"] == "free_recall" for r in rows):
        # Paper M6 score: micro token recall = total hits / total gold tokens.
        total = sum(int(r.get("recall_total") or 0) for r in rows)
        stats["acc"] = sum(int(r.get("recall_hit") or 0) for r in rows) / total if total else 0.0
        if judged:
            jt = sum(int(r.get("judge_total") or 0) for r in judged)
            stats["judge_acc"] = sum(int(r.get("judge_hit") or 0) for r in judged) / jt if jt else 0.0
    if efficiency:
        curve = budget_curve(correct, lengths)
        stats["bauc"] = bauc(correct, lengths)
        stats["acc_at"] = {b: curve[BUDGET_GRID.index(b)] for b in CHECKPOINTS}
        if judged and len(judged) == len(rows):
            stats["judge_bauc"] = bauc([float(r["judge_correct"]) for r in rows], lengths)
    return stats


def aggregate(scored: List[dict]) -> Dict[str, object]:
    tasks: "OrderedDict[str, dict]" = OrderedDict()
    for r in scored:
        t = tasks.setdefault(r["task_code"], {"capability_code": r["capability_code"], "task": r["task"],
                                              "code": r["task_code"], "rows": [], "subtasks": OrderedDict()})
        t["rows"].append(r)
        if r.get("subtask"):
            t["subtasks"].setdefault(r["subtask"], []).append(r)

    task_out, sub_out = [], []
    for code, t in tasks.items():
        eff = t["capability_code"] == "Gs"
        stats = _group_stats(t["rows"], eff)
        stats.update(capability_code=t["capability_code"], task=t["task"], code=code,
                     answer_formats=sorted({r["answer_format"] for r in t["rows"]}))
        task_out.append(stats)
        for name, rows in t["subtasks"].items():
            s = _group_stats(rows, eff)
            s.update(capability_code=t["capability_code"], task_code=code, subtask=name)
            sub_out.append(s)

    cap_out = []
    for code, name in CAPABILITIES.items():
        ts = [t for t in task_out if t["capability_code"] == code]
        if not ts:
            continue
        rows = [r for r in scored if r["capability_code"] == code]
        eff = code == "Gs"
        all_judged = all(t["judge_acc"] is not None for t in ts)
        cap = {
            "code": code,
            "name": name,
            "n": len(rows),
            "n_tasks": len(ts),
            "complete": len(rows) == FULL_TASK_COUNTS[code],
            "pooling": HEADLINE_POOLING[code],
            "acc_macro": mean([t["acc"] for t in ts]),
            "acc_micro": sum(t["acc"] * t["n"] for t in ts) / len(rows),
            "judge_macro": mean([t["judge_acc"] for t in ts]) if all_judged else None,
            "judge_micro": sum(t["judge_acc"] * t["n"] for t in ts) / len(rows) if all_judged else None,
            "headline_metric": "B-AUC" if eff else "ACC",
        }
        if eff:
            cap["bauc_macro"] = mean([t["bauc"] for t in ts])
            cap["judge_bauc_macro"] = mean([t.get("judge_bauc") for t in ts]) \
                if all(t.get("judge_bauc") is not None for t in ts) else None
            cap["headline"] = cap["bauc_macro"]
            cap["headline_judge"] = cap["judge_bauc_macro"]
        else:
            suffix = "micro" if cap["pooling"] == "items" else "macro"
            cap["headline"] = cap[f"acc_{suffix}"]
            cap["headline_judge"] = cap[f"judge_{suffix}"]
        cap_out.append(cap)

    overall = {
        "n": len(scored),
        "headline_macro": mean([c["headline"] for c in cap_out]) if len(cap_out) == len(CAPABILITIES) else None,
        "acc_micro": mean([float(r["correct"]) for r in scored]),
        "errors": sum(1 for r in scored if r.get("error")),
        "gen_seconds": sum(float(r.get("gen_seconds") or 0) for r in scored),
    }
    return {"overall": overall, "capabilities": cap_out, "tasks": task_out, "subtasks": sub_out}


def render_markdown(summary: Dict[str, object], run: Dict[str, object]) -> str:
    caps, tasks, subs, overall = summary["capabilities"], summary["tasks"], summary["subtasks"], summary["overall"]
    judged = any(c.get("headline_judge") is not None for c in caps)
    model = run.get("model", {})
    subset = any(not c["complete"] for c in caps) or len(caps) < len(CAPABILITIES)
    L: List[str] = []
    L.append(f"# RAIL Evaluation Report: {run.get('model_name')}")
    L.append("")
    L.append(f"Generated {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')} · RAIL v2 "
             f"({overall['n']} of 5306 items) · prompt protocol `{run.get('prompt_style')}`")
    L.append("")
    if subset:
        L.append("> **Partial run.** Not every capability was evaluated on its full item set "
                 f"(limit per task: {run.get('limit_per_task') or 'none'}; filters: "
                 f"{run.get('filters') or 'none'}). Scores are for this subset only and are not "
                 "comparable with the full-benchmark numbers in the paper.")
        L.append("")

    L.append("## Run summary")
    L.append("")
    L.append("| Setting | Value |")
    L.append("|---|---|")
    for label, value in [
        ("Model", f"`{model.get('model_path')}`"),
        ("Loader", f"Hugging Face `transformers` {model.get('transformers')} · adapter `{model.get('adapter')}` "
                   f"· `{model.get('model_class')}`"),
        ("Precision / device", f"{model.get('dtype')} on {model.get('gpu') or model.get('device')}"),
        ("Decoding", f"greedy, max_new_tokens = {run.get('max_new_tokens')}"),
        ("Prompt protocol", run.get("prompt_description")),
        ("Multi-clip items", "clips merged into one waveform" if model.get("merge_multi_audio")
         else "clips passed to the model as separate audio inputs"),
        ("Audio input window", f"{model.get('max_clip_seconds'):.0f} s per clip (longer clips are truncated by the processor)"
         if model.get("max_clip_seconds") else "not limited by the processor"),
        ("LLM judge", run.get("judge_model") or "not run (rule-based ACC only)"),
        ("Items / errors", f"{overall['n']} / {overall['errors']}"),
        ("Generation time", f"{overall['gen_seconds'] / 60:.1f} min "
                            f"({overall['gen_seconds'] / max(overall['n'], 1):.2f} s per item)"),
    ]:
        L.append(f"| {label} | {value} |")
    L.append("")

    L.append("## Headline results")
    L.append("")
    L.append("One score per core capability, computed as in the paper's main table: rule-based accuracy (ACC) "
             "for Auditory Processing, Reasoning, Memory and Knowledge, and B-AUC for Processing Efficiency. "
             "Auditory Processing, Reasoning and Knowledge pool all items; Memory and Processing Efficiency "
             "average their task scores. All values are percentages.")
    L.append("")
    head = "| Capability | Code | Metric | Score |" + (" Judge |" if judged else "") + " Pooling | Items | Tasks |"
    L.append(head)
    L.append("|---|---|---|---:|" + ("---:|" if judged else "") + "---|---:|---:|")
    for c in caps:
        row = f"| {c['name']} | {c['code']} | {c['headline_metric']} | {_pct(c['headline'])} |"
        if judged:
            row += f" {_pct(c.get('headline_judge'))} |"
        pooling = "items" if c["pooling"] == "items" else "mean of tasks"
        L.append(row + f" {pooling} | {c['n']} | {c['n_tasks']} |")
    if overall["headline_macro"] is not None:
        L.append(f"| **Mean of the five capabilities** | | | **{_pct(overall['headline_macro'])}** |"
                 + (" |" if judged else "") + f" | {overall['n']} | {len(tasks)} |")
    L.append("")
    gs = next((c for c in caps if c["code"] == "Gs"), None)
    if gs:
        L.append(f"Processing Efficiency answer accuracy without the reasoning budget: "
                 f"{_pct(gs['acc_macro'])} (mean over tasks).")
        L.append("")

    L.append("## Results by task")
    L.append("")
    L.append("| Capability | Task | Code | Items | ACC | B-AUC |" + (" Judge |" if judged else "")
             + " `Answer:` found | Mean reason length |")
    L.append("|---|---|---|---:|---:|---:|" + ("---:|" if judged else "") + "---:|---:|")
    for t in tasks:
        row = (f"| {CAPABILITIES[t['capability_code']]} | {t['task']} | {t['code']} | {t['n']} | "
               f"{_pct(t['acc'])} | {_pct(t.get('bauc'))} |")
        if judged:
            row += f" {_pct(t.get('judge_acc'))} |"
        row += f" {_pct(t['answer_field_rate'], 0)}% | {_num(t['mean_reason_len'])} |"
        L.append(row)
    L.append("")
    L.append("For Free Recall Memory (M6) the ACC column is the token recall rate.")
    L.append("")

    if subs:
        L.append("## Results by subtask")
        L.append("")
        L.append("| Task | Subtask | Items | ACC |" + (" Judge |" if judged else ""))
        L.append("|---|---|---:|---:|" + ("---:|" if judged else ""))
        for s in subs:
            row = f"| {s['task_code']} | {s['subtask']} | {s['n']} | {_pct(s['acc'])} |"
            if judged:
                row += f" {_pct(s.get('judge_acc'))} |"
            L.append(row)
        L.append("")

    eff = [t for t in tasks if t["capability_code"] == "Gs"]
    if eff:
        L.append("## Processing efficiency detail")
        L.append("")
        L.append("Acc≤b is the share of items answered correctly with a reason of at most b tokens. "
                 "B-AUC is the normalised area under this curve for b = 0…50.")
        L.append("")
        L.append("| Code | Task | ACC | " + " | ".join(f"Acc≤{b}" for b in CHECKPOINTS)
                 + " | B-AUC | Reason recovered | Median reason length |")
        L.append("|---|---|---:|" + "---:|" * len(CHECKPOINTS) + "---:|---:|---:|")
        for t in eff:
            L.append(f"| {t['code']} | {t['task']} | {_pct(t['acc'])} | "
                     + " | ".join(_pct(t["acc_at"][b]) for b in CHECKPOINTS)
                     + f" | {_pct(t['bauc'])} | {_pct(t['reason_rate'], 0)}% | {_num(t['median_reason_len'])} |")
        L.append("")

    L.append("## Output diagnostics")
    L.append("")
    n = max(overall["n"], 1)
    answer_field = sum(t["answer_field_rate"] * t["n"] for t in tasks) / n
    reason_rate = sum(t["reason_rate"] * t["n"] for t in tasks) / n
    L.append(f"- Responses with an explicit `Answer:` field: {100 * answer_field:.1f}%. Responses without "
             "one are scored on their last line.")
    L.append(f"- Responses with a recoverable `Reason:` segment: {100 * reason_rate:.1f}% "
             "(items without one cannot meet any B-AUC budget).")
    errors = run.get("error_examples") or []
    L.append(f"- Generation errors: {overall['errors']} (scored as wrong; `infer` retries them when re-run)"
             + (". " + "; ".join(f"`{e['id']}` {e['error']}" for e in errors[:10]) if errors else "") + ".")
    hit = sum(t["hit_max_new_tokens"] for t in tasks)
    L.append(f"- Responses that reached max_new_tokens: {hit}.")
    trunc = [(t["code"], t["truncated_items"]) for t in tasks if t["truncated_items"]]
    if trunc:
        L.append("- Items with at least one clip longer than the model's audio window (truncated): "
                 + ", ".join(f"{c} {k}" for c, k in trunc) + ".")
    unmapped = run.get("closed_label_unmapped")
    if unmapped:
        L.append(f"- Closed-label answers that could not be mapped to an allowed label: {unmapped}.")
    L.append("")

    L.append("## Scoring methodology")
    L.append("")
    L.append("Responses are parsed into a `Reason` and an `Answer` segment. The rules below follow the scripts "
             "used for the paper.")
    L.append("")
    L.append("- **Multiple choice** (Auditory Processing, Reasoning, Memory, Knowledge): the answer may be the "
             "option letter, the letter with its text, or the text alone. Text answers must contain every "
             "gold token in order and no token that belongs only to another option.")
    L.append("- **Closed label** (Processing Efficiency, UM n-back): the answer is mapped to one of the allowed "
             "labels and must equal the gold label.")
    L.append("- **Open-ended math** (RQ, Math Reasoning): numeric equality with the gold value, also accepted "
             "when the number appears inside a short phrase.")
    L.append("- **Free recall** (M6): token recall of the target items (lowercase alphanumeric tokens of "
             "three or more characters, multiset overlap), pooled over items (total hits / total gold tokens).")
    L.append("- **B-AUC** (Processing Efficiency): reason length is the number of `[A-Za-z0-9]+` tokens in the "
             "Reason segment; B-AUC = (1/50) Σ_{b=0}^{49} (Acc≤b + Acc≤b+1)/2.")
    if judged:
        L.append(f"- **Judge**: {run.get('judge_model')} decides whether the full response's final answer matches "
                 "the gold answer (paper prompt, temperature 0); M6 uses the recall variant.")
    L.append("- Capability scores: item-pooled for Auditory Processing, Reasoning and Knowledge; mean of task "
             "scores for Memory and Processing Efficiency (the paper's convention). Both variants are in "
             "`summary.json`.")
    L.append("")
    return "\n".join(L)
