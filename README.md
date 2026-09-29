# RAIL project page

Source of the project page for **RAIL: Rethinking Auditory Intelligence in Large Audio-Language Models with a CHC-Grounded Benchmark** (NeurIPS 2026), served at <https://aims-rail.github.io/RAIL/>.

- Paper: [arXiv:2606.11260](https://arxiv.org/abs/2606.11260) · [Hugging Face](https://huggingface.co/papers/2606.11260)
- Dataset: [AIMS-RAIL/RAIL](https://huggingface.co/datasets/AIMS-RAIL/RAIL)
- Code: [AIMS-RAIL/RAIL](https://github.com/AIMS-RAIL/RAIL) (`main` branch)

## Editing

The page is one static file, `index.html`, with figures in `assets/figures/` and no build step. Open it in a browser to preview, or serve the folder locally:

```bash
python3 -m http.server 8765
```

All benchmark numbers live in JavaScript arrays near the top of the `<script>` block (`CAPS`, `MODELS`, `HU`, `CMP`), so the tables, leaderboard and radar chart update together when results change.

GitHub Pages publishes this branch (`gh-pages`, root folder). `.nojekyll` keeps GitHub from running Jekyll on it.
