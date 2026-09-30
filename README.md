# RAIL: Rethinking Auditory Intelligence in Large Audio-Language Models with a CHC-Grounded Benchmark

**NeurIPS 2026**

Hongyu Jin<sup>1,\*</sup>, Siyi Wang<sup>1,\*</sup>, Yang Xiao<sup>1,\*</sup>, Jiaheng Dong<sup>1,\*</sup>, Shihong Tan<sup>4</sup>, Kaiyuan Peng<sup>1</sup>, Georgiana Juravle<sup>2</sup>, Shanquan Chen<sup>3</sup>, Gongping Huang<sup>4</sup>, Hong Jia<sup>5</sup>, Eun-Jung Holden<sup>1</sup>, James Bailey<sup>6</sup>, Ting Dang<sup>1,†</sup>

<sup>1</sup>The University of Melbourne, <sup>2</sup>Alexandru Ioan Cuza University of Iași, <sup>3</sup>The University of Hong Kong, <sup>4</sup>Wuhan University, <sup>5</sup>The University of Auckland, <sup>6</sup>Monash University

<sup>\*</sup>Equal contribution. <sup>†</sup>Corresponding author.

Contact: Hongyu Jin (hongyuj1@student.unimelb.edu.au), Ting Dang (ting.dang@unimelb.edu.au)

[Project page](https://aims-rail.github.io/RAIL/) · [arXiv](https://arxiv.org/abs/2606.11260) · [Paper on Hugging Face](https://huggingface.co/papers/2606.11260) · [Dataset on Hugging Face](https://huggingface.co/datasets/AIMS-RAIL/RAIL)

**TL;DR:** RAIL evaluates large audio-language models the way cognitive science assesses human listeners. Grounded in Cattell–Horn–Carroll (CHC) theory, it organizes 5,306 audio questions into five core auditory capabilities and 32 subcapabilities, and compares 26 LALMs with a human baseline. Models do well on knowledge inherited from text pretraining, while fine-grained auditory perception and processing efficiency remain below human level.

![RAIL: five CHC capabilities and 32 subcapabilities](https://aims-rail.github.io/RAIL/assets/figures/chc_structure.webp)

| Capability | CHC | Subcapabilities | Samples | Audio (h) |
|---|---|---|---|---|
| Auditory Processing | Ga | 7 | 1,170 | 4.93 |
| Fluid Reasoning | Gf | 3 | 322 | 3.49 |
| Memory | Gsm + Glr | 6 | 1,000 | 13.0 |
| Processing Efficiency | Gs + Gt | 9 | 1,800 | 2.01 |
| Acquired Knowledge | Gc + Gkn | 7 | 1,014 | 7.2 |
| **Total** | | **32** | **5,306** | **30.6** |

## Repository layout

```
data/<capability>/<TASK>.jsonl    32 task files, 5,306 items
audio/<capability>/<TASK>/        6,485 audio clips (Git LFS)
eval/                             evaluation toolkit: HF inference, scoring, English report
scripts/build_rail_v2.py          rebuilds data/ and audio/ from the v1 release
metadata.json                     Croissant metadata with RAI fields
build_report.json                 build checks, statistics and flagged items
```

Clone with Git LFS to get the audio:

```bash
git lfs install
git clone https://github.com/AIMS-RAIL/RAIL.git
```

## Benchmark structure

Items are organised in three levels that follow the paper: 5 core capabilities (CHC broad
abilities), 32 tasks (CHC narrow abilities, named and coded as in the paper) and, where a task
has more than one design, subtasks. Each task is one file, `data/<capability>/<TASK>.jsonl`
(`U1/U9` is stored as `U1_U9`).

| Capability | Task (code) | Items | Answer format | Subtasks (items) |
|---|---|---:|---|---|
| Auditory Processing (Ga) | Speech Sound Discrimination (`US`) | 153 | multiple choice | Emotional Prosody Comparison (55); Confusable Word Identification (58); Minimal-Pair Discrimination (40) |
|  | Resistance to Auditory Stimulus Distortion (`UR`) | 181 | multiple choice | Babble Noise (65); Band-Limiting (40); Background Music (36); Reverberation (40) |
|  | Maintaining and Judging Rhythm (`U8`) | 160 | multiple choice | Beat Regularity Detection (40); Meter Identification (38); Tempo Change Detection (30); Tempo Comparison (52) |
|  | Absolute Pitch (`UP`) | 200 | multiple choice | MIDI Pitch Identification (80); Note Name Identification (50); Relative Pitch Comparison (40); Frequency Estimation (30) |
|  | Musical Discrimination and Judgment (`U1/U9`) | 102 | multiple choice | Aesthetic and Emotional Judgment (8); Genre Recognition (35); Instrument Identification (50); Musical Structure Analysis (9) |
|  | Sound Localization (`UL`) | 182 | multiple choice | Direction Identification (82); Azimuth Estimation (50); Distance Estimation (30); Motion Trajectory (20) |
|  | Phonetic Coding (`PC`) | 192 | multiple choice | Isolated Phoneme Identification (42); Concurrent Phoneme Segregation (50); Staggered-Onset Phoneme Identification (50); Stereo-Separated Phoneme Identification (50) |
| Reasoning (Gf) | Induction (`I`) | 68 | multiple choice | Pattern Abstraction (34); Rule Induction (34) |
|  | General Sequential Reasoning (`RG`) | 154 | multiple choice | Sequential Reasoning with General Rules (124); Auditory Cognitive Puzzle (30) |
|  | Quantitative Reasoning (`RQ`) | 100 | open-ended, multiple choice | Math Reasoning (50); Counting (50) |
| Memory (Gsm/Glr) | Memory Span (`MS`) | 150 | multiple choice | — |
|  | Associative Memory (`MA`) | 150 | multiple choice | — |
|  | Meaningful Memory (`MM`) | 150 | multiple choice | Multi-Turn Dialogue (100); Narrated Passage (50) |
|  | Free Recall Memory (`M6`) | 200 | free recall | — |
|  | Memory for Sound Patterns (`UM`) | 200 | closed label, multiple choice | N-Back (70); Speaker Tracking (70); Prosodic Contour Matching (60) |
|  | Working Memory (`WM`) | 150 | multiple choice | — |
| Processing Efficiency (Gs) | Perceptual Speed (`P`) | 200 | closed label | — |
|  | Rate-of-Test-Taking (`R9`) | 200 | closed label | — |
|  | Number Facility (`N`) | 200 | closed label | — |
|  | Reading Speed (`RS`) | 200 | closed label | — |
|  | Simple Reaction Time (`R1`) | 200 | closed label | — |
|  | Choice Reaction Time (`R2`) | 200 | closed label | — |
|  | Semantic Processing Speed (`R4`) | 200 | closed label | — |
|  | Mental Comparison Speed (`R7`) | 200 | closed label | — |
|  | Inspection Time (`IT`) | 200 | closed label | — |
| Knowledge (Gc/Gkn) | General (Verbal) Information (`K0`) | 143 | multiple choice | — |
|  | Language Development (`LD`) | 181 | multiple choice | — |
|  | Listening Ability (`LS`) | 150 | multiple choice | — |
|  | Foreign Language Proficiency (`KL`) | 193 | multiple choice | — |
|  | Geography Achievement (`A5`) | 65 | multiple choice | — |
|  | Mechanical Knowledge (`MK`) | 141 | multiple choice | Anomaly Detection (47); Machine Identification (47); Machine Function Inference (47) |
|  | Knowledge of Behavioral Content (`BC`) | 141 | multiple choice | — |

Capability folders: `auditory_processing`, `reasoning`, `memory`, `processing_efficiency`, `knowledge`.

## Data format

Every line of a task file is one item:

| Field | Description |
|---|---|
| `id` | Unique item id, `<capability>-<task>-<nnnn>` (e.g. `ga-ul-0001`). |
| `capability`, `capability_code` | Core capability and CHC code (`Ga`, `Gf`, `Gsm/Glr`, `Gs`, `Gc/Gkn`). |
| `task`, `task_code` | Task (CHC narrow ability) and its code. |
| `subtask` | Task design within the task; `null` when the task has a single design. |
| `answer_format` | `multiple_choice`, `closed_label` (answer is one of `choices`, no letters shown), `open_ended` or `free_recall`. |
| `question` | Question stem without options or output instructions. |
| `choices` | Option texts without letter prefixes, or the allowed output labels for `closed_label`. |
| `answer` | Gold answer text; for `free_recall`, one target item per line. |
| `answer_label` | Gold option letter for `multiple_choice`, otherwise `null`. |
| `prompt` | Full model-facing prompt used in the paper's evaluation. |
| `audio`, `audio_paths`, `num_audio` | First clip, all clips in presentation order, number of clips (paths relative to the repository root). |
| `duration_s` | Total audio duration of the item in seconds. |
| `source_id` | Item id in the source manifest. |
| `legacy_id`, `legacy_config` | Id and config name in v1, for mapping earlier results. |
| `metadata_json` | Original source row as JSON text (generation parameters, conditions, provenance). |

```json
{"id": "gsm-ma-0001", "capability": "Memory", "task": "Associative Memory", "task_code": "MA",
 "answer_format": "multiple_choice", "question": "Which word means 'sync data'?",
 "choices": ["noik", "niok", "noem", "slaiel"], "answer": "noik", "answer_label": "A",
 "audio_paths": ["audio/memory/MA/gsm-ma-0001.wav"], "duration_s": 17.491, "...": "..."}
```

Loading a clone with `datasets`:

```python
import sys
from datasets import load_dataset

sys.path.insert(0, "eval")
from rail_eval.data import hf_features  # explicit schema: some files have empty `choices`

rail = load_dataset("json", data_files="data/*/*.jsonl", features=hf_features(), split="train")
memory = load_dataset("json", data_files="data/memory/*.jsonl", features=hf_features(), split="train")
```

## Evaluating a model

`eval/` runs a Hugging Face audio-language model on every item, scores each item with the rule
for its answer format, and writes an English report. The prompts and scoring rules are the ones
used for the paper; re-scoring the paper's Qwen2-Audio-7B-Instruct outputs with this code
reproduces the published numbers.

```bash
pip install torch transformers librosa numpy          # openai only for the optional LLM judge
cd eval

# full benchmark on one GPU: inference, scoring, report
python run_rail_eval.py all --model Qwen/Qwen2-Audio-7B-Instruct --out runs/qwen2_audio

# smoke test with 5 items per task
python run_rail_eval.py all --model Qwen/Qwen2-Audio-7B-Instruct --out runs/smoke --limit-per-task 5

# re-score existing predictions, optionally with the GPT-5.4 judge
python run_rail_eval.py score --out runs/qwen2_audio --judge-model gpt-5.4
```

- **Models** are loaded with `transformers` `from_pretrained` (Hub id or local checkpoint).
  Qwen2-Audio has a dedicated adapter; other models go through a generic chat-template adapter
  (`--adapter generic`, `--trust-remote-code`, `--merge-multi-audio` for single-audio models).
- **Prompting**: every prompt is wrapped so the model answers `Reason: ...; Answer: ...`
  (reason capped at 20 words, except Reasoning items); decoding is greedy.
- **Scoring**: strict option matching for multiple choice, label matching for closed-label
  tasks, numeric matching for open-ended math, token recall for free recall (M6), and B-AUC
  over reason length (0–50 tokens) for Processing Efficiency. Capability scores are pooled as
  in the paper's main table.
- **Outputs**: `predictions*.jsonl`, `scored.jsonl`, `summary.json` and `report.md` in the run folder.

Runs resume after interruption and can be sharded over GPUs (`--num-shards`, `--shard-index`).
A Slurm template is in `eval/slurm/`. See [eval/README.md](eval/README.md) for details. The
original scoring scripts used for the paper are kept in `eval/legacy/`.

## Rebuilding the data

`data/` and `audio/` are generated from the v1 release (commit `8ef268a`) by
`scripts/build_rail_v2.py`, which relabels items with the paper's taxonomy, normalises the
answer fields, rebuilds the evaluated memory prompts and checks the result against the paper's
statistics:

```bash
git worktree add ../rail-v1 8ef268a
python scripts/build_rail_v2.py --v1-root ../rail-v1 --out /path/to/release
```

The output folder also contains the Hugging Face dataset card (`README.md`).

## Changes from v1

- Items are labelled `capability` → `task` → `subtask` as in the paper; the v1 fields
  `benchmark`, `subset`, `task` and `ability` are replaced, and files are split by task instead of
  by source manifest.
- Every `id` is unique (v1 had 35 duplicates); v1 ids are kept in `legacy_id`.
- Memory prompts and options are the exact ones used in evaluation (v1 lacked the Memory Span
  prompts, the options of Working and Meaningful Memory prompts, and the prosodic-matching
  options); speaker-tracking items use the evaluated wording.
- Options carry no letter prefixes; Processing Efficiency items list their allowed labels.
- `duration_s` is measured from the audio files for every item.
- Audio is stored as `audio/<capability>/<TASK>/<id>[_k].<ext>`.
- The evaluation toolkit in `eval/` replaces the standalone scoring scripts.

## Citation

```bibtex
@inproceedings{jin2026rail,
  title     = {RAIL: Rethinking Auditory Intelligence in Large Audio-Language Models
               with a CHC-Grounded Benchmark},
  author    = {Jin, Hongyu and Wang, Siyi and Xiao, Yang and Dong, Jiaheng and
               Tan, Shihong and Peng, Kaiyuan and Juravle, Georgiana and Chen, Shanquan and
               Huang, Gongping and Jia, Hong and Holden, Eun-Jung and Bailey, James and
               Dang, Ting},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2606.11260}
}
```

## License

The dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Some items
are derived from existing corpora; use them within the terms of those sources.
