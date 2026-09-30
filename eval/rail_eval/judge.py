"""Optional LLM-as-judge (paper Appendix "LLM-as-Judge Protocol"; judge model GPT-5.4).

Needs the `openai` package and OPENAI_API_KEY (or --judge-base-url for a compatible server).
Verdicts are cached in <run>/judge_cache.jsonl so re-scoring does not call the API again.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

JUDGE_SYSTEM = (
    "You are a strict and impartial evaluator for benchmark outputs.\n\n"
    "Instructions:\n"
    "1. Identify the final answer expressed in the model response.\n"
    "2. The final answer may be expressed directly, indirectly, or as a paraphrase.\n"
    "3. Compare the model's final answer with the gold answer for semantic equivalence.\n"
    "4. Do not grade reasoning quality. Only judge whether the final answer matches the gold answer.\n\n"
    "Respond with EXACTLY one word — nothing else:\n"
    "  true   — the model's final answer matches the gold answer\n"
    "  false  — the model's final answer does not match, or no definite answer was given"
)
JUDGE_USER = (
    "Gold answer:      {reference}\n"
    "Model response:   {prediction}\n\n"
    "Does the model's final answer match the gold answer?\n"
    "Reply with exactly one word: true or false."
)
RECALL_SYSTEM = (
    "You are a strict evaluator for item-list recall tasks.\n\n"
    "You will receive:\n1) a list of gold items\n2) a model response\n\n"
    "Goal:\n- Compute token-level recall with semantic tolerance:\n"
    "  - allow paraphrase/synonyms/minor spelling errors if meaning is clear\n"
    "  - do not give credit for unrelated content\n\n"
    "Return JSON only, exactly in this schema:\n{\"hit\": <integer>, \"total\": <integer>}\n\n"
    "Where:\n- total = total number of meaningful gold tokens\n- hit   = number of recalled gold tokens\n"
)
RECALL_USER = (
    "Gold items (one per line):\n{items}\n\n"
    "Model response:\n{prediction}\n\n"
    "Return JSON only: {{\"hit\": <int>, \"total\": <int>}}."
)
_VERDICT_RE = re.compile(r"\b(true|false)\b", re.IGNORECASE)


def _key(item_id: str, response: str, model: str) -> str:
    return hashlib.sha1(f"{model}\x00{item_id}\x00{response}".encode("utf-8")).hexdigest()


class Judge:
    def __init__(self, model: str = "gpt-5.4", base_url: Optional[str] = None, workers: int = 8,
                 cache_path: Optional[Path] = None):
        import openai

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key and not base_url:
            raise RuntimeError("LLM judge requested but OPENAI_API_KEY is not set")
        self.client = openai.OpenAI(api_key=api_key or "EMPTY", base_url=base_url)
        self.model = model
        self.workers = workers
        self.cache_path = cache_path
        self.cache: Dict[str, dict] = {}
        if cache_path and cache_path.exists():
            for line in cache_path.open(encoding="utf-8"):
                if line.strip():
                    rec = json.loads(line)
                    self.cache[rec["key"]] = rec

    def _chat(self, system: str, user: str, max_tokens: int) -> str:
        rsp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            max_completion_tokens=max_tokens,
        )
        return (rsp.choices[0].message.content or "").strip()

    def _judge_one(self, row: dict) -> dict:
        response = str(row.get("response") or "").strip()
        if not response:
            return {"judge_correct": 0.0, "judge_raw": "", "judge_error": "empty response"}
        try:
            if row["answer_format"] == "free_recall":
                from .scoring import split_items

                items = "\n".join(f"- {x}" for x in split_items(row["gold"]))
                raw = self._chat(RECALL_SYSTEM, RECALL_USER.format(items=items, prediction=response), 120)
                data = json.loads(raw)
                total = max(int(data.get("total", 0)), 0) or int(row.get("recall_total") or 0)
                hit = min(max(int(data.get("hit", 0)), 0), total)
                return {"judge_correct": hit / total if total else 0.0, "judge_hit": hit, "judge_total": total,
                        "judge_raw": raw, "judge_error": None}
            raw = self._chat(JUDGE_SYSTEM, JUDGE_USER.format(reference=row["gold"], prediction=response), 8)
            m = _VERDICT_RE.search(raw)
            return {"judge_correct": float(bool(m) and m.group(1).lower() == "true"), "judge_raw": raw,
                    "judge_error": None}
        except Exception as exc:  # network / parse errors are recorded, item counts as wrong
            return {"judge_correct": 0.0, "judge_raw": "", "judge_error": str(exc)[:300]}

    def run(self, rows: List[dict]) -> None:
        todo = []
        for row in rows:
            key = _key(row["id"], str(row.get("response") or ""), self.model)
            cached = self.cache.get(key)
            if cached and not cached.get("judge_error"):
                row.update({k: cached[k] for k in ("judge_correct", "judge_hit", "judge_total", "judge_raw",
                                                   "judge_error") if k in cached})
            else:
                todo.append((key, row))
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            results = list(pool.map(lambda kr: self._judge_one(kr[1]), todo))
        handle = self.cache_path.open("a", encoding="utf-8") if self.cache_path else None
        for (key, row), res in zip(todo, results):
            row.update(res)
            if handle:
                handle.write(json.dumps({"key": key, "id": row["id"], **res}, ensure_ascii=False) + "\n")
        if handle:
            handle.close()
