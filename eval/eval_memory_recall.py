"""
evaluation_metrices_recall.py
=============================
Token-level recall evaluation for free-recall tasks (e.g., m6 item lists).

This script keeps the "strict / llm / both" workflow style from
evaluation_metrices.py, but uses ONLY token-level item-list recall:

Strict (rule-based, deterministic)
  - strict_recall_score: multiset token overlap / gold token total

LLM-as-judge (semantic token-level)
  - llm_recall_score: judge returns token overlap hit/total with semantic tolerance

Input fields (per record):
  - answer or ground_truth: gold reference text (item list)
  - model_output (preferred) or answer_text/response_text/response_full: model output
  - ok (optional): if ok=false, record is skipped

Optional:
  - gold_payload.target_facts: if present, used as canonical gold item list
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


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_FACT_SPLIT_RE = re.compile(r"(?:^|\n)\s*(?:[-*]\s+|\d+\.\s+)?")


def _tokenize(text: str, min_len: int = 3) -> list[str]:
    return [t for t in _TOKEN_RE.findall(str(text or "").lower()) if len(t) >= min_len]


def _split_lines_as_items(text: str) -> list[str]:
    parts = _FACT_SPLIT_RE.split(str(text or "").strip())
    return [p.strip() for p in parts if p and p.strip()]


def _extract_gold_items(item: dict) -> list[str]:
    gp = item.get("gold_payload") or {}
    tf = gp.get("target_facts")
    if isinstance(tf, list):
        items = [str(x).strip() for x in tf if str(x).strip()]
        if items:
            return items

    ref = str(
        item.get("answer")
        or item.get("ground_truth")
        or item.get("answer_gt")
        or ""
    ).strip()
    if not ref:
        return []
    return _split_lines_as_items(ref)


def _load_manifest_gold_map(manifest_path: pathlib.Path) -> dict[str, dict]:
    """
    Build a map: item_id -> {"answer": str, "target_facts": list[str]}

    Supports:
      - entries with item_id directly, or
      - entries with numeric id (mapped to m6_{id})
    """
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"manifest must be a JSON list: {manifest_path}")

    out: dict[str, dict] = {}
    for o in data:
        if not isinstance(o, dict):
            continue
        iid = o.get("item_id")
        if isinstance(iid, str) and iid.strip():
            key = iid.strip()
        elif "id" in o:
            try:
                key = f"m6_{int(o['id'])}"
            except Exception:
                continue
        else:
            continue

        answer = str(o.get("answer") or "").strip()
        tf = o.get("target_facts")
        if isinstance(tf, list):
            target_facts = [str(x).strip() for x in tf if str(x).strip()]
        else:
            target_facts = _split_lines_as_items(answer)
        out[key] = {"answer": answer, "target_facts": target_facts}
    return out


def _extract_prediction_text(item: dict) -> str:
    return str(
        item.get("model_output")
        or item.get("response_full")
        or item.get("response_text")
        or item.get("response")
        or item.get("answer_label")
        or item.get("answer_text")
        or ""
    ).strip()


def _counter_overlap(a_tokens: list[str], b_tokens: list[str]) -> int:
    if not a_tokens or not b_tokens:
        return 0
    ac: dict[str, int] = {}
    bc: dict[str, int] = {}
    for tok in a_tokens:
        ac[tok] = ac.get(tok, 0) + 1
    for tok in b_tokens:
        bc[tok] = bc.get(tok, 0) + 1
    hit = 0
    for tok, cnt in ac.items():
        if tok in bc:
            hit += min(cnt, bc[tok])
    return hit


def strict_recall_score(item: dict) -> dict:
    gold_items = _extract_gold_items(item)
    pred_text = _extract_prediction_text(item)
    pred_items = _split_lines_as_items(pred_text)

    pred_tokens: list[str] = []
    for it in pred_items:
        pred_tokens.extend(_tokenize(it))

    gold_tokens: list[str] = []
    for it in gold_items:
        gold_tokens.extend(_tokenize(it))

    token_hit = _counter_overlap(pred_tokens, gold_tokens)
    token_total = len(gold_tokens)
    token_recall = (token_hit / token_total) if token_total else 0.0

    return {
        "token_hit": token_hit,
        "token_total": token_total,
        "token_recall": token_recall,
        "gold_items": gold_items,
    }


_RECALL_JUDGE_SYSTEM = """\
You are a strict evaluator for item-list recall tasks.

You will receive:
1) a list of gold items
2) a model response

Goal:
- Compute token-level recall with semantic tolerance:
  - allow paraphrase/synonyms/minor spelling errors if meaning is clear
  - do not give credit for unrelated content

Return JSON only, exactly in this schema:
{"hit": <integer>, "total": <integer>}

Where:
- total = total number of meaningful gold tokens
- hit   = number of recalled gold tokens
"""


_RECALL_JUDGE_USER_TMPL = (
    "Gold items (one per line):\n{items}\n\n"
    "Model response:\n{prediction}\n\n"
    "Return JSON only: {{\"hit\": <int>, \"total\": <int>}}."
)


def llm_recall_score(
    item: dict,
    *,
    client=None,
    model: str = "gpt-5.4",
) -> dict:
    import os
    try:
        import openai as _openai
    except ImportError as exc:
        return {"token_hit": 0, "token_total": 0, "token_recall": 0.0, "error": str(exc)}

    pred = _extract_prediction_text(item)
    gold_items = _extract_gold_items(item)
    gold_tokens: list[str] = []
    for it in gold_items:
        gold_tokens.extend(_tokenize(it))
    if not pred or not gold_tokens:
        return {
            "token_hit": 0,
            "token_total": len(gold_tokens),
            "token_recall": 0.0,
            "error": "empty prediction or missing gold items",
        }

    if client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return {
                "token_hit": 0,
                "token_total": len(gold_tokens),
                "token_recall": 0.0,
                "error": "OPENAI_API_KEY not set and no client provided",
            }
        client = _openai.OpenAI(api_key=api_key)

    items_block = "\n".join(f"- {f}" for f in gold_items)
    prompt = _RECALL_JUDGE_USER_TMPL.format(items=items_block, prediction=pred)

    hit = 0
    total = len(gold_tokens)
    try:
        rsp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _RECALL_JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_completion_tokens=120,
        )
        raw = rsp.choices[0].message.content.strip()
        data = json.loads(raw)
        hit = int(data.get("hit", 0))
        parsed_total = int(data.get("total", total))
        if parsed_total > 0:
            total = parsed_total
        if hit < 0:
            hit = 0
        if hit > total:
            hit = total
    except Exception as exc:
        return {
            "token_hit": hit,
            "token_total": total,
            "token_recall": (hit / total) if total else 0.0,
            "error": str(exc),
        }
    return {
        "token_hit": hit,
        "token_total": total,
        "token_recall": (hit / total) if total else 0.0,
        "error": None,
    }


def load_records(path: pathlib.Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    data = json.loads(text)
    return data if isinstance(data, list) else [data]


def _save_records(records: list[dict], out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate_file(
    path: pathlib.Path,
    *,
    group_by: str = "task",
    mode: str = "strict",
    llm_client=None,
    llm_model: str = "gpt-5.4",
    judge_results_dir: Optional[pathlib.Path] = None,
    gold_map: Optional[dict[str, dict]] = None,
) -> dict:
    records = load_records(path)
    do_strict = mode in ("strict", "both")
    do_llm = mode in ("llm", "both")

    total = 0
    strict_token_hit = strict_token_total = 0
    llm_token_hit = llm_token_total = 0

    per_group: dict[str, dict] = defaultdict(lambda: {
        "total": 0,
        "strict_token_hit": 0,
        "strict_token_total": 0,
        "llm_token_hit": 0,
        "llm_token_total": 0,
    })

    active_records: list[dict] = []
    for item in records:
        if not item.get("ok", True):
            continue
        if gold_map is not None:
            iid = str(item.get("item_id") or "").strip()
            gm = gold_map.get(iid)
            if gm:
                item["answer"] = gm["answer"]
                gp = item.get("gold_payload")
                if not isinstance(gp, dict):
                    gp = {}
                gp["target_facts"] = gm["target_facts"]
                gp["target_item_count"] = len(gm["target_facts"])
                item["gold_payload"] = gp
        total += 1
        active_records.append(item)
        grp = str(item.get(group_by, "unknown"))
        per_group[grp]["total"] += 1

        if do_strict:
            s = strict_recall_score(item)
            item["strict_token_hit"] = s["token_hit"]
            item["strict_token_total"] = s["token_total"]
            item["strict_recall_score"] = round(s["token_recall"], 6)

            strict_token_hit += s["token_hit"]
            strict_token_total += s["token_total"]

            per_group[grp]["strict_token_hit"] += s["token_hit"]
            per_group[grp]["strict_token_total"] += s["token_total"]

    if do_llm and active_records:
        n_workers = min(32, len(active_records))

        def _judge_item(item):
            return item, llm_recall_score(item, client=llm_client, model=llm_model)

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = [pool.submit(_judge_item, it) for it in active_records]
            done = 0
            for fut in as_completed(futs):
                item, j = fut.result()
                grp = str(item.get(group_by, "unknown"))
                item["LLM-as-judge_token_hit"] = j["token_hit"]
                item["LLM-as-judge_token_total"] = j["token_total"]
                item["LLM-as-judge_recall_score"] = round(j["token_recall"], 6)
                if j["error"]:
                    item["LLM-as-judge_error"] = j["error"]

                llm_token_hit += j["token_hit"]
                llm_token_total += j["token_total"]
                per_group[grp]["llm_token_hit"] += j["token_hit"]
                per_group[grp]["llm_token_total"] += j["token_total"]

                done += 1
                if done % 20 == 0 or done == len(active_records):
                    print(
                        f"  [LLM recall judge] {done}/{len(active_records)} items done "
                        f"(token_hit={llm_token_hit}, token_total={llm_token_total})",
                        flush=True,
                    )

    if judge_results_dir is not None:
        _save_records(records, judge_results_dir / path.name)

    out = {"file": path.name, "total": total}
    if do_strict:
        out["strict_token_hit"] = strict_token_hit
        out["strict_token_total"] = strict_token_total
        out["strict_recall_score"] = (
            strict_token_hit / strict_token_total if strict_token_total else 0.0
        )
    if do_llm:
        out["llm_token_hit"] = llm_token_hit
        out["llm_token_total"] = llm_token_total
        out["llm_recall_score"] = llm_token_hit / llm_token_total if llm_token_total else 0.0

    for g in per_group.values():
        if do_strict:
            g["strict_recall_score"] = (
                g["strict_token_hit"] / g["strict_token_total"] if g["strict_token_total"] else 0.0
            )
        if do_llm:
            g["llm_recall_score"] = (
                g["llm_token_hit"] / g["llm_token_total"] if g["llm_token_total"] else 0.0
            )
    out["per_group"] = dict(per_group)
    return out


def evaluate_directory(
    directory: pathlib.Path,
    *,
    group_by: str = "task",
    mode: str = "strict",
    llm_client=None,
    llm_model: str = "gpt-5.4",
    judge_results_dir: Optional[pathlib.Path] = None,
    gold_map: Optional[dict[str, dict]] = None,
) -> list[dict]:
    paths = sorted(directory.glob("*.json")) + sorted(directory.glob("*.jsonl"))
    results = []
    for p in paths:
        try:
            out_dir = None if judge_results_dir is None else (judge_results_dir / (p.stem))
            results.append(
                evaluate_file(
                    p,
                    group_by=group_by,
                    mode=mode,
                    llm_client=llm_client,
                    llm_model=llm_model,
                    judge_results_dir=out_dir,
                    gold_map=gold_map,
                )
            )
        except Exception as exc:
            print(f"[WARN] Could not evaluate {p.name}: {exc}", file=sys.stderr)
    return results


def _print_result(r: dict) -> None:
    print(f"\n{'─'*60}")
    print(f"File : {r['file']}")
    print(f"n    : {r['total']}")
    if "strict_recall_score" in r:
        print(
            "Strict token-recall : "
            f"{r['strict_token_hit']}/{r['strict_token_total']} = {r['strict_recall_score']:.4f}"
        )
    if "llm_recall_score" in r:
        print(
            "LLM token-recall    : "
            f"{r['llm_token_hit']}/{r['llm_token_total']} = {r['llm_recall_score']:.4f}"
        )


def _write_scores_json(results: list[dict], scores_json: pathlib.Path) -> None:
    rows = []
    for r in results:
        row = {"file": r["file"], "n": r["total"]}
        for k in (
            "strict_recall_score",
            "llm_recall_score",
            "strict_token_hit",
            "strict_token_total",
            "llm_token_hit",
            "llm_token_total",
        ):
            if k in r:
                row[k] = r[k]
        rows.append(row)
    scores_json.parent.mkdir(parents=True, exist_ok=True)
    scores_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[scores] Wrote {scores_json} ({len(rows)} rows)")


def main() -> None:
    import os
    parser = argparse.ArgumentParser(description="Recall-oriented evaluation metrics.")
    parser.add_argument("path", help="Result JSON/JSONL file or directory.")
    parser.add_argument("--group", default="task", help="Field to group by.")
    parser.add_argument("--mode", default="strict", choices=["strict", "llm", "both"])
    parser.add_argument("--llm-model", default="gpt-5.4")
    parser.add_argument("--judge-results-dir", default="", help="Where to save annotated outputs.")
    parser.add_argument("--scores-json", default="", help="Where to save summary JSON.")
    parser.add_argument(
        "--gold-manifest",
        default="",
        help="Optional JSON manifest containing canonical gold answers/items "
             "(e.g., data/manifest_m6.json). When set, gold is overridden by item_id.",
    )
    args = parser.parse_args()

    llm_client = None
    if args.mode in ("llm", "both"):
        try:
            import openai as _openai
        except ImportError:
            print("[ERROR] openai package not installed. Run: pip install openai", file=sys.stderr)
            sys.exit(1)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("[ERROR] OPENAI_API_KEY is not set.", file=sys.stderr)
            sys.exit(1)
        llm_client = _openai.OpenAI(api_key=api_key)

    p = pathlib.Path(args.path)
    out_dir = pathlib.Path(args.judge_results_dir) if args.judge_results_dir else None
    scores_json = pathlib.Path(args.scores_json) if args.scores_json else None
    gold_map = None
    if args.gold_manifest:
        manifest_path = pathlib.Path(args.gold_manifest)
        gold_map = _load_manifest_gold_map(manifest_path)
        print(f"[gold] loaded {len(gold_map)} items from {manifest_path}")

    if p.is_dir():
        results = evaluate_directory(
            p,
            group_by=args.group,
            mode=args.mode,
            llm_client=llm_client,
            llm_model=args.llm_model,
            judge_results_dir=out_dir,
            gold_map=gold_map,
        )
        for r in results:
            _print_result(r)
        if results:
            tot_n = sum(r["total"] for r in results)
            print(f"\n{'='*60}")
            print(f"TOTAL across {len(results)} files  n={tot_n}")
            if "strict_recall_score" in results[0]:
                sh = sum(r["strict_token_hit"] for r in results)
                st = sum(r["strict_token_total"] for r in results)
                print(f"  Strict token-recall : {sh}/{st} = {(sh/st if st else 0):.4f}")
            if "llm_recall_score" in results[0]:
                lh = sum(r["llm_token_hit"] for r in results)
                lt = sum(r["llm_token_total"] for r in results)
                print(f"  LLM token-recall    : {lh}/{lt} = {(lh/lt if lt else 0):.4f}")
            if scores_json is not None:
                _write_scores_json(results, scores_json)
    elif p.is_file():
        r = evaluate_file(
            p,
            group_by=args.group,
            mode=args.mode,
            llm_client=llm_client,
            llm_model=args.llm_model,
            judge_results_dir=out_dir,
            gold_map=gold_map,
        )
        _print_result(r)
        if scores_json is not None:
            _write_scores_json([r], scores_json)
    else:
        print(f"[ERROR] Path not found: {p}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
