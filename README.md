---
configs:
- config_name: all
  data_files:
  - split: test
    path: data/all.jsonl
- config_name: reasoning__general_rule
  data_files:
  - split: test
    path: data/reasoning__general_rule.jsonl
- config_name: reasoning__cognitive_puzzles
  data_files:
  - split: test
    path: data/reasoning__cognitive_puzzles.jsonl
- config_name: reasoning__quantitative_reasoning_prompted
  data_files:
  - split: test
    path: data/reasoning__quantitative_reasoning_prompted.jsonl
- config_name: reasoning__induction_dataset_mcq_update
  data_files:
  - split: test
    path: data/reasoning__induction_dataset_mcq_update.jsonl
- config_name: auditory__auditory_manifest_all
  data_files:
  - split: test
    path: data/auditory__auditory_manifest_all.jsonl
- config_name: knowledge__knowledge_manifest_all
  data_files:
  - split: test
    path: data/knowledge__knowledge_manifest_all.jsonl
- config_name: efficiency__presentation_benchmark_9x200_bench1800
  data_files:
  - split: test
    path: data/efficiency__presentation_benchmark_9x200_bench1800.jsonl
- config_name: memory__ma
  data_files:
  - split: test
    path: data/memory__ma.jsonl
- config_name: memory__m6
  data_files:
  - split: test
    path: data/memory__m6.jsonl
- config_name: memory__ms
  data_files:
  - split: test
    path: data/memory__ms.jsonl
- config_name: memory__mw_voice_editing_preview
  data_files:
  - split: test
    path: data/memory__mw_voice_editing_preview.jsonl
- config_name: memory__um
  data_files:
  - split: test
    path: data/memory__um.jsonl
- config_name: memory__mm_combination
  data_files:
  - split: test
    path: data/memory__mm_combination.jsonl
---

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
| Fluid Reasoning | Gf | 3 | 322 | 5.19 |
| Memory | Gsm + Glr | 6 | 1,000 | 13.0 |
| Processing Efficiency | Gs + Gt | 9 | 1,800 | 2.01 |
| Acquired Knowledge | Gc + Gkn | 7 | 1,014 | 7.2 |
| **Total** | | **32** | **5,306** | **30.6** |

## Dataset files

This folder is generated for direct Hugging Face dataset upload. Each row uses relative audio paths rooted at this repository folder.

### NeurIPS / Croissant Metadata

- `metadata.json` is a Croissant-style metadata file with core fields and minimal RAI fields.
- `metadata.json` includes the Hugging Face dataset URL, CC-BY-4.0 license URL, checksums, and RAI fields.
- `build_summary.json` contains build counts and skipped-source diagnostics.

### Schema

- `id`: stable row id.
- `benchmark`, `subset`, `task`, `subtask`, `ability`: benchmark grouping fields.
- `ability`: normalized CHC-style ability label from the benchmark taxonomy.
- `question`, `prompt`, `choices`, `answer`, `answer_label`: evaluation fields.
- For multiple-choice questions, predictions should be counted as correct if they match either `answer` or `answer_label`.
- `audio`: first relative audio path for convenience.
- `audio_paths`: all relative audio paths needed by the item.
- `metadata_json`: sanitized source row as a JSON string; audio path fields are relative or null.

### Counts

- Total rows: 5306
- Unique audio files linked/copied: 6187
- Rows with no primary audio: 0

### Subsets

- `reasoning__general_rule`: 124 rows, `reasoning__general_rule.jsonl`
- `reasoning__cognitive_puzzles`: 30 rows, `reasoning__cognitive_puzzles.jsonl`
- `reasoning__quantitative_reasoning_prompted`: 100 rows, `reasoning__quantitative_reasoning_prompted.jsonl`
- `reasoning__induction_dataset_mcq_update`: 68 rows, `reasoning__induction_dataset_mcq_update.jsonl`
- `auditory__auditory_manifest_all`: 1170 rows, `auditory__auditory_manifest_all.jsonl`
- `knowledge__knowledge_manifest_all`: 1014 rows, `knowledge__knowledge_manifest_all.jsonl`
- `efficiency__presentation_benchmark_9x200_bench1800`: 1800 rows, `efficiency__presentation_benchmark_9x200_bench1800.jsonl`
- `memory__ma`: 150 rows, `memory__ma.jsonl`
- `memory__m6`: 200 rows, `memory__m6.jsonl`
- `memory__ms`: 150 rows, `memory__ms.jsonl`
- `memory__mw_voice_editing_preview`: 150 rows, `memory__mw_voice_editing_preview.jsonl`
- `memory__um`: 200 rows, `memory__um.jsonl`
- `memory__mm_combination`: 150 rows, `memory__mm_combination.jsonl`

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
