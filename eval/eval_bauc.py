#!/usr/bin/env python3
"""Compute budgeted area-under-curve (B-AUC) from scored item-level rows.

This script deliberately does not score model answers. It assumes each row is
already scored with a correctness/score column and has one budget column, such
as reason_length for token-budget B-AUC or net_processing_time_sec for time
B-AUC.

At each budget B:
    curve(B) = sum(score_i for rows with budget_i <= B) / N

Rows with missing budget values remain in N but cannot contribute under finite
budgets. This matches the existing reason-budget scripts in this repository.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_BUDGETS = list(range(0, 51))
CHECKPOINT_BUDGETS = [0, 5, 10, 15, 20, 30, 40, 50]
TRUE_STRINGS = {"1", "true", "t", "yes", "y", "correct"}
FALSE_STRINGS = {"0", "false", "f", "no", "n", "incorrect", "wrong"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute normalized budgeted AUC from item-level CSV/JSONL rows. "
            "Use --budget-col reason_length for model token B-AUC, or a time "
            "column such as net_processing_time_sec for time B-AUC."
        )
    )
    parser.add_argument("--input", required=True, nargs="+", help="Input CSV or JSONL file(s).")
    parser.add_argument(
        "--format",
        choices=("auto", "csv", "jsonl"),
        default="auto",
        help="Input format. Default: infer from extension.",
    )
    parser.add_argument(
        "--score-col",
        default="correct",
        help="Column/field containing bool or numeric score in [0,1]. Default: correct.",
    )
    parser.add_argument(
        "--budget-col",
        default="reason_length",
        help="Column/field containing budget value. Default: reason_length.",
    )
    parser.add_argument(
        "--group-by",
        nargs="*",
        default=[],
        help="Optional columns to group by, e.g. --group-by display_name ability_code.",
    )
    parser.add_argument(
        "--budgets",
        default="0:50:1",
        help=(
            "Budget grid. Use start:end:step inclusive, or comma-separated values. "
            "Examples: 0:50:1, 0:60:1, 0,5,10,20,30,40,50."
        ),
    )
    parser.add_argument(
        "--checkpoints",
        default="0,5,10,15,20,30,40,50",
        help="Comma-separated checkpoints to include as columns in the summary CSV.",
    )
    parser.add_argument(
        "--exclude-budget-gt",
        type=float,
        default=None,
        help="Drop rows whose budget value is greater than this threshold before computing N.",
    )
    parser.add_argument(
        "--exclude-missing-budget",
        action="store_true",
        help="Drop rows with missing/invalid budget values before computing N.",
    )
    parser.add_argument(
        "--out-summary",
        default=None,
        help="Output summary CSV path. Default: print summary CSV to stdout.",
    )
    parser.add_argument(
        "--out-curves",
        default=None,
        help="Optional output curves JSON path.",
    )
    return parser.parse_args()


def parse_budget_grid(spec: str) -> list[float]:
    spec = spec.strip()
    if not spec:
        raise ValueError("Empty budget grid")
    if ":" in spec:
        parts = [float(part) for part in spec.split(":")]
        if len(parts) != 3:
            raise ValueError("--budgets range must be start:end:step")
        start, end, step = parts
        if step <= 0:
            raise ValueError("--budgets step must be positive")
        values = []
        current = start
        epsilon = step / 1_000_000.0
        while current <= end + epsilon:
            values.append(round(current, 10))
            current += step
        return values
    return [float(part.strip()) for part in spec.split(",") if part.strip()]


def format_budget(value: float) -> str:
    if math.isfinite(value) and value.is_integer():
        return str(int(value))
    return ("%g" % value).replace(".", "p")


def normalized_auc(xs: list[float], ys: list[float]) -> float:
    if not xs or len(xs) != len(ys):
        return float("nan")
    if len(xs) == 1:
        return float(ys[0])
    total = 0.0
    for i in range(len(xs) - 1):
        width = xs[i + 1] - xs[i]
        total += width * (ys[i] + ys[i + 1]) / 2.0
    span = xs[-1] - xs[0]
    return total / span if span > 0 else float(ys[0])


def parse_score(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        if math.isnan(float(value)):
            return None
        return float(value)
    text = str(value).strip().lower()
    if text == "":
        return None
    if text in TRUE_STRINGS:
        return 1.0
    if text in FALSE_STRINGS:
        return 0.0
    try:
        numeric = float(text)
    except ValueError:
        return None
    if math.isnan(numeric):
        return None
    return numeric


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        return None if math.isnan(numeric) else numeric
    text = str(value).strip()
    if text == "":
        return None
    try:
        numeric = float(text)
    except ValueError:
        return None
    return None if math.isnan(numeric) else numeric


def detect_format(path: Path, explicit: str) -> str:
    if explicit != "auto":
        return explicit
    if path.suffix.lower() == ".jsonl":
        return "jsonl"
    return "csv"


def iter_records(path: Path, input_format: str) -> list[dict[str, Any]]:
    fmt = detect_format(path, input_format)
    if fmt == "jsonl":
        records = []
        with path.open() as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(f"{path}:{line_no} is not a JSON object")
                records.append(obj)
        return records

    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def group_key(record: dict[str, Any], group_by: list[str]) -> tuple[str, ...]:
    if not group_by:
        return ("all",)
    return tuple(str(record.get(col, "")) for col in group_by)


def build_groups(
    records: list[dict[str, Any]],
    group_by: list[str],
    score_col: str,
    budget_col: str,
    exclude_budget_gt: float | None,
    exclude_missing_budget: bool,
) -> tuple[dict[tuple[str, ...], list[dict[str, float | None]]], int, int, int]:
    groups: dict[tuple[str, ...], list[dict[str, float | None]]] = defaultdict(list)
    skipped_bad_score = 0
    skipped_missing_budget = 0
    skipped_budget_gt = 0

    for record in records:
        score = parse_score(record.get(score_col))
        if score is None:
            skipped_bad_score += 1
            continue
        budget = parse_float(record.get(budget_col))
        if budget is None:
            if exclude_missing_budget:
                skipped_missing_budget += 1
                continue
        elif exclude_budget_gt is not None and budget > exclude_budget_gt:
            skipped_budget_gt += 1
            continue
        groups[group_key(record, group_by)].append({"score": score, "budget": budget})

    return groups, skipped_bad_score, skipped_missing_budget, skipped_budget_gt


def summarize_group(key: tuple[str, ...], rows: list[dict[str, float | None]], budgets: list[float]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        curve = [0.0 for _ in budgets]
        mean_score = 0.0
        coverage = 0.0
    else:
        mean_score = sum(float(row["score"]) for row in rows) / n
        coverage = sum(1 for row in rows if row["budget"] is not None) / n
        curve = [
            sum(
                float(row["score"])
                for row in rows
                if row["budget"] is not None and float(row["budget"]) <= budget
            )
            / n
            for budget in budgets
        ]

    return {
        "key": key,
        "n_items": n,
        "mean_score": mean_score,
        "budget_coverage": coverage,
        "bauc": normalized_auc(budgets, curve),
        "curve": [{"budget": budget, "score": score} for budget, score in zip(budgets, curve)],
    }


def write_summary(
    path: Path | None,
    summaries: list[dict[str, Any]],
    group_by: list[str],
    checkpoints: list[float],
) -> None:
    group_cols = group_by or ["group"]
    fieldnames = [
        *group_cols,
        "n_items",
        "mean_score",
        "budget_coverage",
        "bauc",
        *[f"score_budget_le_{format_budget(budget)}" for budget in checkpoints],
    ]

    output_handle = path.open("w", newline="") if path else None
    try:
        handle = output_handle if output_handle is not None else None
        if handle is None:
            import sys

            handle = sys.stdout
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            row = {
                "n_items": summary["n_items"],
                "mean_score": summary["mean_score"],
                "budget_coverage": summary["budget_coverage"],
                "bauc": summary["bauc"],
            }
            for col, value in zip(group_cols, summary["key"]):
                row[col] = value
            curve_lookup = {point["budget"]: point["score"] for point in summary["curve"]}
            for budget in checkpoints:
                if budget in curve_lookup:
                    value = curve_lookup[budget]
                else:
                    value = None
                    for point in summary["curve"]:
                        if point["budget"] <= budget:
                            value = point["score"]
                        else:
                            break
                    if value is None:
                        value = 0.0
                row[f"score_budget_le_{format_budget(budget)}"] = value
            writer.writerow(row)
    finally:
        if output_handle is not None:
            output_handle.close()


def write_curves(path: Path, summaries: list[dict[str, Any]], group_by: list[str], meta: dict[str, Any]) -> None:
    group_cols = group_by or ["group"]
    payload = {"meta": meta, "groups": []}
    for summary in summaries:
        group = {col: value for col, value in zip(group_cols, summary["key"])}
        payload["groups"].append(
            {
                **group,
                "n_items": summary["n_items"],
                "mean_score": summary["mean_score"],
                "budget_coverage": summary["budget_coverage"],
                "bauc": summary["bauc"],
                "curve": summary["curve"],
            }
        )
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    budgets = parse_budget_grid(args.budgets)
    checkpoints = parse_budget_grid(args.checkpoints)
    records = []
    for input_path in args.input:
        records.extend(iter_records(Path(input_path), args.format))

    groups, skipped_bad_score, skipped_missing_budget, skipped_budget_gt = build_groups(
        records=records,
        group_by=args.group_by,
        score_col=args.score_col,
        budget_col=args.budget_col,
        exclude_budget_gt=args.exclude_budget_gt,
        exclude_missing_budget=args.exclude_missing_budget,
    )
    summaries = [summarize_group(key, rows, budgets) for key, rows in groups.items()]
    summaries.sort(key=lambda row: (*row["key"],))

    out_summary = Path(args.out_summary) if args.out_summary else None
    write_summary(out_summary, summaries, args.group_by, checkpoints)

    if args.out_curves:
        write_curves(
            Path(args.out_curves),
            summaries,
            args.group_by,
            {
                "inputs": args.input,
                "score_col": args.score_col,
                "budget_col": args.budget_col,
                "budgets": budgets,
                "skipped_bad_score": skipped_bad_score,
                "skipped_missing_budget": skipped_missing_budget,
                "skipped_budget_gt": skipped_budget_gt,
            },
        )


if __name__ == "__main__":
    main()
