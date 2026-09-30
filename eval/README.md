# RAIL evaluation

Runs a Hugging Face audio-language model on every RAIL item, scores each item with the rule for
its answer format, and writes an English report (`report.md`) plus machine-readable results.
The prompt wrappers and scoring rules are the ones used for the paper; re-scoring the paper's
Qwen2-Audio-7B-Instruct outputs with this code reproduces its published numbers (see below).

## Quick start

```bash
# one GPU, full benchmark (5,306 items), inference + scoring + report
python run_rail_eval.py all --model Qwen/Qwen2-Audio-7B-Instruct --out runs/qwen2_audio

# smoke test: 5 random items per task (160 items)
python run_rail_eval.py all --model Qwen/Qwen2-Audio-7B-Instruct --out runs/smoke --limit-per-task 5

# Slurm (one GPU); see the header of slurm/rail_eval.sbatch for all variables
sbatch --account=<account> --partition=<gpu-partition> \
  --export=ALL,MODEL=Qwen/Qwen2-Audio-7B-Instruct,OUT=$PWD/runs/qwen2_audio,PYTHON=$(which python) \
  slurm/rail_eval.sbatch
```

A full run of Qwen2-Audio-7B-Instruct takes about 40 minutes on one A100 80GB (0.4 s per item).

`--data` defaults to the RAIL folder that contains this directory; a Hub id (`AIMS-RAIL/RAIL`) also works.
Requirements: `torch`, `transformers`, `librosa`, `numpy`; `openai` only for the LLM judge.

## Commands

| Command | What it does |
|---|---|
| `infer` | Generates a response for every selected item and appends it to `predictions*.jsonl`. Re-running resumes; failed items are retried. |
| `score` | Merges all prediction files, scores them, optionally runs the LLM judge, writes `scored.jsonl`, `summary.json`, `report.md`. |
| `all` | `infer` followed by `score`. |

Useful options: `--tasks UL U1_U9 M6`, `--capabilities Gs Memory`, `--limit-per-task N`,
`--max-new-tokens 512`, `--prompt-style paper|direct`, `--num-shards K --shard-index i`
(then `score` once after all shards finish), `--judge-model gpt-5.4` (needs `OPENAI_API_KEY`).

## Models

Models are loaded with `from_pretrained` from a Hub id or a local checkpoint.

| Adapter | Models | Notes |
|---|---|---|
| `qwen2_audio` | Qwen2-Audio-7B(-Instruct) | Multiple clips are passed as separate audio inputs. |
| `generic` | Processors that accept audio in `apply_chat_template` (e.g. Qwen2.5-Omni, Gemma-3n, Voxtral, Audio Flamingo 3 in recent `transformers`) | Tries `AutoModelForImageTextToText`, `AutoModelForCausalLM`, `AutoModelForSeq2SeqLM`, `AutoModel`. |
| `auto` (default) | | `qwen2_audio` when `model_type == "qwen2_audio"`, else `generic`. |

Use `--merge-multi-audio` for models that accept only one audio input: the clips of an item
(up to five) are concatenated with 0.3 s of silence and the prompt says so. To support another
model family, subclass `AudioLM` in `rail_eval/models.py` and add it to `ADAPTERS`.

## Evaluation protocol

**Prompting.** Every item's `prompt` is wrapped so the model answers in one line,
`Reason: ...; Answer: ...`. Reasoning items (Gf) allow open reasoning; all other items cap the
reason at 20 words (`--reason-max-words`) and list the allowed labels when there are at most 16.
Decoding is greedy.

**Scoring.**

| Answer format | Used by | Rule |
|---|---|---|
| `multiple_choice` | most tasks | The parsed answer may be a letter, a letter with text, or text. Text must contain the gold tokens in order and no token that only occurs in other options. |
| `closed_label` | Processing Efficiency, UM n-back | Answer mapped to an allowed label; must equal the gold label. |
| `open_ended` | RQ Math Reasoning | Numeric equality (also inside a short phrase). |
| `free_recall` | M6 | Token recall of target items (tokens of ≥3 characters), pooled over items. |

Processing Efficiency is summarised by **B-AUC**: the fraction of items answered correctly with a
reason of at most *b* tokens (`[A-Za-z0-9]+`), averaged by the trapezoidal rule over b = 0…50.
The optional **LLM judge** uses the paper's GPT-5.4 prompt on the full response.

**Capability scores** follow the paper's main table: item-pooled accuracy for Auditory Processing,
Reasoning and Knowledge; the mean of task scores for Memory and Processing Efficiency (B-AUC).

## Outputs

| File | Content |
|---|---|
| `run_config.json` | Model, loader, decoding and selection settings. |
| `predictions*.jsonl` | Prompt sent, raw response, token counts, timing, truncation, errors. |
| `scored.jsonl` | Per item: parsed reason/answer, reason length, correctness, gold. |
| `summary.json` | Scores per capability, task and subtask (micro and macro), B-AUC curves. |
| `report.md` | The English report. |

## Consistency with the paper

Re-scoring the paper's Qwen2-Audio-7B-Instruct outputs with `rail_eval.scoring`:

| | Paper | This code |
|---|---:|---:|
| Processing Efficiency ACC / B-AUC | 46.56 / 42.32 | 46.50 / 42.27 |
| Memory MS / WM / UM / MA / MM / M6 | 44.00 / 34.70 / 35.00 / 70.67 / 46.00 / 18.07 | 44.00 / 34.67 / 35.00 / 71.33 / 46.67 / 18.07 |
| Reasoning Induction / QR / Sequential | 44.12 / 24.00 / 31.82 | 44.12 / 24.00 / 31.82 |

Auditory Processing and Knowledge could not be re-scored because their source predictions are
not accessible; they use the same multiple-choice rule.
