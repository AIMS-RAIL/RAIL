#!/usr/bin/env python3
"""Run a Hugging Face audio-language model on RAIL, score it, and write an English report.

  python run_rail_eval.py all   --model Qwen/Qwen2-Audio-7B-Instruct --out runs/qwen2_audio
  python run_rail_eval.py infer --model /path/to/ckpt --out runs/x --num-shards 4 --shard-index 0
  python run_rail_eval.py score --out runs/x [--judge-model gpt-5.4]

`infer` appends to <out>/predictions*.jsonl and resumes where it stopped. `score` merges all
prediction shards, scores every item with the rule for its answer format, optionally runs the
LLM judge, and writes scored.jsonl, summary.json and report.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rail_eval.data import count_by_task, load_items  # noqa: E402
from rail_eval.prompts import build_prompt  # noqa: E402
from rail_eval.report import aggregate, render_markdown  # noqa: E402
from rail_eval.scoring import gold_reference, score_item  # noqa: E402

DEFAULT_DATA = str(Path(__file__).resolve().parents[1])
PROMPT_DESCRIPTIONS = {
    "paper": "paper wrapper: `Reason: ...; Answer: ...`; reasoning capped at {w} words with the allowed "
             "labels listed (Ga, Gsm/Glr, Gs, Gc/Gkn); uncapped reasoning for Gf",
    "direct": "item prompt only, no Reason/Answer wrapper (B-AUC is not meaningful)",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def add_selection_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data", default=DEFAULT_DATA, help="RAIL v2 folder or Hub dataset id")
    p.add_argument("--tasks", nargs="*", help="task codes to keep, e.g. UL U1_U9 M6")
    p.add_argument("--capabilities", nargs="*", help="capability codes or names, e.g. Gs Memory")
    p.add_argument("--limit-per-task", type=int, default=0, help="seeded random subset per task (0 = all)")
    p.add_argument("--seed", type=int, default=0)


def add_infer_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, help="Hub id or local checkpoint directory")
    p.add_argument("--model-name", help="display name in the report (default: last path component)")
    p.add_argument("--adapter", default="auto", help="auto | qwen2_audio | generic")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--attn-implementation", default=None)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--merge-multi-audio", action="store_true",
                   help="concatenate multi-clip items into one waveform (for single-audio models)")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--prompt-style", default="paper", choices=sorted(PROMPT_DESCRIPTIONS))
    p.add_argument("--reason-max-words", type=int, default=20)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--log-every", type=int, default=25)


def add_score_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--judge-model", default=None, help="e.g. gpt-5.4; needs OPENAI_API_KEY")
    p.add_argument("--judge-base-url", default=None)
    p.add_argument("--judge-workers", type=int, default=8)


def prediction_path(out: Path, args) -> Path:
    if args.num_shards > 1:
        return out / f"predictions.shard{args.shard_index:02d}of{args.num_shards:02d}.jsonl"
    return out / "predictions.jsonl"


def cmd_infer(args) -> None:
    from rail_eval.models import build_model

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items = load_items(args.data, args.tasks, args.capabilities, args.limit_per_task, args.seed)
    items = items[args.shard_index::args.num_shards]
    pred_file = prediction_path(out, args)
    done = set()
    if pred_file.exists():
        for line in pred_file.open(encoding="utf-8"):
            if line.strip():
                rec = json.loads(line)
                if not rec.get("error"):
                    done.add(rec["id"])
    todo = [it for it in items if it["id"] not in done]
    log(f"{len(items)} items selected ({len(done)} already done, {len(todo)} to run) -> {pred_file}")
    if not todo:
        return

    model = build_model(args.model, args.adapter, dtype=args.dtype, device=args.device,
                        trust_remote_code=args.trust_remote_code,
                        attn_implementation=args.attn_implementation,
                        merge_multi_audio=args.merge_multi_audio)
    config = {
        "model_name": args.model_name or Path(args.model.rstrip("/")).name,
        "model": model.describe(),
        "prompt_style": args.prompt_style,
        "prompt_description": PROMPT_DESCRIPTIONS[args.prompt_style].format(w=args.reason_max_words),
        "reason_max_words": args.reason_max_words,
        "max_new_tokens": args.max_new_tokens,
        "limit_per_task": args.limit_per_task,
        "seed": args.seed,
        "filters": {k: v for k, v in (("tasks", args.tasks), ("capabilities", args.capabilities)) if v},
        "data": str(args.data),
        "selected_items_by_task": count_by_task(load_items(args.data, args.tasks, args.capabilities,
                                                           args.limit_per_task, args.seed)),
    }
    (out / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"loaded {config['model']['model_class']} via adapter {config['model']['adapter']} "
        f"on {config['model']['gpu'] or args.device}")

    start = time.time()
    with pred_file.open("a", encoding="utf-8") as handle:
        for i, item in enumerate(todo, 1):
            prompt = build_prompt(item, args.prompt_style, args.reason_max_words)
            rec = {"id": item["id"], "task_code": item["task_code"], "prompt_sent": prompt}
            try:
                rec.update(model.generate(item["audio_files"], prompt, args.max_new_tokens))
                rec["error"] = None
            except Exception as exc:  # keep going; the item is scored as wrong and reported
                rec.update(response="", error=f"{type(exc).__name__}: {exc}"[:500])
                if "out of memory" in str(exc).lower():
                    import torch

                    torch.cuda.empty_cache()
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            handle.flush()
            if i % args.log_every == 0 or i == len(todo):
                rate = (time.time() - start) / i
                log(f"{i}/{len(todo)} done · {rate:.2f} s/item · eta {rate * (len(todo) - i) / 60:.1f} min · "
                    f"last {item['id']}: {str(rec.get('response'))[:80]!r}")


def cmd_score(args) -> None:
    out = Path(args.out)
    items = {it["id"]: it for it in load_items(args.data)}
    preds = {}
    for f in sorted(out.glob("predictions*.jsonl")):
        for line in f.open(encoding="utf-8"):
            if line.strip():
                rec = json.loads(line)
                if rec["id"] not in preds or preds[rec["id"]].get("error"):
                    preds[rec["id"]] = rec  # later successful retries replace errors
    if not preds:
        raise SystemExit(f"no predictions found in {out}")

    scored = []
    for pid, rec in preds.items():
        item = items[pid]
        row = {
            "id": pid,
            "capability_code": item["capability_code"],
            "capability": item["capability"],
            "task_code": item["task_code"],
            "task": item["task"],
            "subtask": item["subtask"],
            "answer_format": item["answer_format"],
            "gold": gold_reference(item),
            "response": rec.get("response", ""),
            "error": rec.get("error"),
        }
        for key in ("n_input_tokens", "n_output_tokens", "hit_max_new_tokens", "gen_seconds", "n_clips_truncated"):
            row[key] = rec.get(key)
        row.update(score_item(item, row["response"]))
        scored.append(row)
    order = {pid: i for i, pid in enumerate(items)}
    scored.sort(key=lambda r: order[r["id"]])

    run = json.loads((out / "run_config.json").read_text(encoding="utf-8")) if (out / "run_config.json").exists() else {}
    if args.judge_model:
        from rail_eval.judge import Judge

        log(f"LLM judge {args.judge_model} on {len(scored)} items")
        Judge(args.judge_model, args.judge_base_url, args.judge_workers, out / "judge_cache.jsonl").run(scored)
        run["judge_model"] = args.judge_model
    run["closed_label_unmapped"] = sum(1 for r in scored
                                       if r["answer_format"] == "closed_label" and not r.get("answer_normalized"))
    run["error_examples"] = [{"id": r["id"], "error": str(r["error"]).split(".")[0][:120]}
                             for r in scored if r.get("error")]

    summary = aggregate(scored)
    with (out / "scored.jsonl").open("w", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(json.dumps({"run": run, **summary}, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    (out / "report.md").write_text(render_markdown(summary, run), encoding="utf-8")
    log(f"scored {len(scored)} items -> {out / 'report.md'}")
    for c in summary["capabilities"]:
        log(f"  {c['name']:22s} {c['headline_metric']:5s} {100 * (c['headline'] or 0):6.2f}  (n={c['n']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("infer", "score", "all"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True, help="run directory")
        add_selection_args(p)
        if name in ("infer", "all"):
            add_infer_args(p)
        if name in ("score", "all"):
            add_score_args(p)
    args = parser.parse_args()
    if args.command in ("infer", "all"):
        cmd_infer(args)
    if args.command in ("score", "all"):
        if args.command == "all" and getattr(args, "num_shards", 1) > 1:
            log("sharded run: score after all shards finish with `score --out ...`")
            return
        cmd_score(args)


if __name__ == "__main__":
    main()
