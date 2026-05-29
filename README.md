# Why Far Looks Up: Probing Spatial Representation in Vision-Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2605.30161-b31b1b.svg)](https://arxiv.org/abs/2605.30161)
[![Project Page](https://img.shields.io/badge/Project%20Page-online-blue.svg)](https://cheolhong0916.github.io/whyfarlooksup.github.io/)
[![SpatialTunnel](https://img.shields.io/badge/SpatialTunnel-dataset-green.svg)](https://github.com/cube-c/spatialtunnel-dataset-gen)

A lightweight framework for diagnosing **how vision-language models internally
represent spatial relations** — *left / right*, *above / below*, *far / close* —
beyond what benchmark accuracy reveals.

It implements the **contrastive probing** methodology from our paper *"Why
Far Looks Up: Probing Spatial Representation in Vision-Language Models"*: by
swapping the two objects inside a spatial question and measuring the difference
between the resulting hidden states (Δ vectors), we obtain a *representation-level*
view of how cleanly a model encodes a spatial axis, and to what extent
vertical and distance directions are coupled (the perspective heuristic
"higher in the image ⇒ farther away").

> **TL;DR.** Two text-only forwards per image (one question + its swapped
> version), one forward hook on every transformer layer, and a handful of
> numpy operations on the resulting Δ vectors give you axis-coherence curves,
> 6×6 delta-similarity heatmaps, 2D/3D PCA plots, and a complementary
> VD-Entanglement Index for any VLM you can load with 🤗 `transformers`.

---

## What this code does

For every spatial question in [EmbSpatial-Bench](https://github.com/mengfeidu/EmbSpatial-Bench),
the script builds a **minimal contrastive pair** by swapping `obj1 ↔ obj2`:

```
Original: "Is the [cup] to the left or right of the [book]?"   → left
Swapped : "Is the [book] to the left or right of the [cup]?"   → right
```

It then runs both questions through the VLM, captures the **last-token hidden
state of every transformer layer**, and forms the per-sample delta:

$$\Delta r = h(\text{swapped}) - h(\text{original})$$

These Δ vectors are the basic unit of analysis. From them the script computes:

| Output | What it measures |
|---|---|
| **Axis Coherence** (per group, per layer) | How consistently the model uses a single direction in hidden space to encode an axis (horizontal / vertical / distance). Mean pairwise cosine similarity of sign-corrected Δ's. |
| **VD-EI** (Vertical–Distance Entanglement Index) | Directional coupling between vertical and distance representations — i.e., potential coupling associated with the perspective heuristic. Complementary diagnostic; computed offline from the saved 6×6 matrix. |
| **6×6 Δ-similarity heatmap** (per layer) | Cosine similarity between the mean Δ of each category — diagonal block structure means clean axis separation. |
| **2D + 3D PCA plots** (per layer) | Visualisation of embeddings and Δ vectors, coloured by category / group. |

Everything is saved as `.npz` / `.csv` / `.json` so you can re-plot or run new
analyses without re-running inference.

---

## Release

| Component | Link |
|---|---|
| Contrastive probing code (this repo) | [cheolhong0916/contrastive-probing](https://github.com/cheolhong0916/contrastive-probing) |
| Project page | [whyfarlooksup](https://cheolhong0916.github.io/whyfarlooksup.github.io/) |
| Paper (arXiv) | [arXiv:2605.30161](https://arxiv.org/abs/2605.30161) |
| EmbSpatial-Bench TSV (dataset) | [ch-min/EmbSpatial-Bench-tsv](https://huggingface.co/datasets/ch-min/EmbSpatial-Bench-tsv) |
| SpatialTunnel dataset generation | [cube-c/spatialtunnel-dataset-gen](https://github.com/cube-c/spatialtunnel-dataset-gen) |
| Fine-tuned checkpoints (80k / 400k / 800k / 2M) | [HF Collection](https://huggingface.co/collections/ch-min/why-far-looks-up-data-scale-fine-tuned-checkpoints) |

Exact HF repo IDs for each scale are registered in `probing.py`
(`MODEL_REGISTRY`).

---

## Supported models

The framework ships with extractor wrappers for the following VLM families.
Adding a new family is two short methods + one registry entry (see
[Adding a new model](#adding-a-new-model)).

| Registry key | Model family | HF default checkpoint |
|---|---|---|
| `qwen25` | Qwen2.5-VL | `Qwen/Qwen2.5-VL-3B-Instruct` |
| `qwen3` | Qwen3-VL (32B / 235B-MoE) | `Qwen/Qwen3-VL-32B-Instruct` |
| `molmo` | Molmo-7B (Allen AI) | `allenai/Molmo-7B-O-0924` |
| `molmo2` | Molmo2-8B | `allenai/Molmo2-8B` |
| `nvila` | NVILA / NVILA-Lite | `Efficient-Large-Model/NVILA-Lite-2B` |

---

## Installation

The framework needs only the standard scientific Python stack plus
🤗 `transformers`. Per-model dependencies (e.g. `qwen-vl-utils`) are imported
lazily — you only need the ones for the models you actually probe.

Minimal install:

```bash
pip install torch transformers pillow numpy pandas tqdm \
            matplotlib seaborn scikit-learn
```

Model-specific extras:

```bash
pip install qwen-vl-utils                # qwen25, qwen3
# nvila: clone https://github.com/NVlabs/VILA and add to PYTHONPATH
```

A pinned environment we have tested with is provided in
[docs/requirements.lock.txt](docs/requirements.lock.txt).

---

## Data

The default benchmark is **EmbSpatial-Bench** (TSV format, base64-encoded
images). The TSV is mirrored on the Hugging Face Hub at
[`ch-min/EmbSpatial-Bench-tsv`](https://huggingface.co/datasets/ch-min/EmbSpatial-Bench-tsv):

```bash
huggingface-cli download ch-min/EmbSpatial-Bench-tsv \
    EmbSpatial-Bench.tsv --repo-type dataset --local-dir ./data
```

Then point `--data_path ./data/EmbSpatial-Bench.tsv` (this is also the
script's default). The loader normalises category aliases (`under` → `below`)
and filters `Unknown` distance answers automatically.

You can use any other benchmark by producing a TSV with the same columns
(`index, question_id, category, question, image, answer, A, B, C, D`).

### SpatialTunnel benchmark

**SpatialTunnel** is the Blender-rendered diagnostic benchmark introduced in
the paper to isolate spatial-shortcut biases by removing background and
perspective confounds. The generation code lives in a separate repository:

  👉 [`cube-c/spatialtunnel-dataset-gen`](https://github.com/cube-c/spatialtunnel-dataset-gen)

It provides two rendering pipelines that drive Blender (5.1.1+) via its
bundled Python:

| Script | What it varies | Output |
|---|---|---|
| `run_phase_variation.sh` | Object angular positions on a 16×16 grid (`--num 12` variants) | up to 3,072 renders + `log.csv` |
| `run_size_variation.sh`  | Object size pairs (`Obj1` 0.1–0.3, `Obj2 = 0.4 − Obj1`, 11 steps, `--num 100` variants) | up to 1,100 renders + `log.csv` |

See that repository's README for Blender installation, CUDA flags, and per-
script options.

---

## Quick start

### 1. Run probing for one checkpoint

```bash
python probing.py \
    --model_type qwen25 \
    --scales vanilla \
    --data_path /path/to/EmbSpatial-Bench.tsv
```

This will:

1. Load the model on `cuda`.
2. Register a forward hook on **every** transformer layer.
3. Run 2 × N forwards (original + swapped) for up to 200 pairs per category.
4. Save raw vectors, axis coherence, delta heatmaps, and per-layer plots under
   `results/saved_data/<model>_<scale>/`.

Useful flags:

| Flag | Default | What it does |
|---|---|---|
| `--scales s1 s2 ...` | all registered | Which checkpoints to run |
| `--max-samples-per-category N` | `200` | Cap per category (`0` = no cap) |
| `--device cuda:0` | `cuda` | Target device |

### 2. Cross-scale / cross-model comparison

After running each checkpoint separately, regenerate comparison plots from
saved data — no model loads needed:

```bash
python probing.py --model_type qwen25 --merge
```

### 3. Multi-GPU shell wrappers

Helper scripts in [scripts/](scripts/) launch all scales of one family in
parallel on assigned GPUs, e.g. `scripts/run_qwen.sh`.

---

## Output layout

```
results/
  saved_data/
    qwen25_vanilla/
      npz/  vectors_vanilla.npz                 # raw embeddings + Δ vectors per layer
      csv/  delta_similarity_vanilla_L<k>.csv   # 6×6 matrices, one per layer
      json/ axis_coherence_vanilla.json         # mean / std / n per (group, layer)
      plots/
        axis_coherence/...
        heatmap/...
        pca/...
        pca_3d/...
  compare/<group>/plots/axis_coherence/cross_scale.png
  logs/<model>_<scale>.log
```

The `.npz` files are the canonical artefact — every downstream analysis
(VD-EI, custom probes, alternative dimensionality reductions, etc.) can be
built directly from them.

---

## Adding a new model

Two steps. Inherit from `BaseHiddenStateExtractor` and implement four small
methods, then register the class.

```python
@register_model(
    model_type   = "my_vlm",
    checkpoints  = {"base": "org/my-vlm-7b"},
    display_name = "MyVLM-7B",
    plot_color   = "#17becf",
)
class MyVLMExtractor(BaseHiddenStateExtractor):
    def _load_model(self):
        self.model     = AutoModelForXxx.from_pretrained(self.model_path, ...)
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def _get_num_layers(self) -> int:
        return len(self.model.language_model.model.layers)

    def _get_layer_module(self, layer_idx: int):
        return self.model.language_model.model.layers[layer_idx]

    def extract_and_predict(self, image, question):
        self.hidden_states = {}                       # reset per-call
        inputs = self.processor(images=image, text=question, return_tensors="pt")
        out    = self.model.generate(**inputs.to(self.device), max_new_tokens=20)
        answer = self.processor.tokenizer.decode(out[0], skip_special_tokens=True)
        return self.hidden_states.copy(), answer      # hooks fill this in
```

Hook registration, cleanup, and prefill-only filtering are handled by the
base class. The shared helper `_find_llm_layers(...)` covers most "language
backbone is buried under several attribute paths" cases automatically.

Then:

```bash
python probing.py --model_type my_vlm --scales base
```

---

## Citation

If you use this code or methodology, please cite our paper.

```bibtex
@article{min2026whyfarlooksup,
  title   = {Why Far Looks Up: Probing Spatial Representation in Vision-Language Models},
  author  = {Min, Cheolhong and Jung, Jaeyun and Lee, Daeun and Jeon, Hyeonseong and
             Su, Yu and Tremblay, Jonathan and Song, Chan Hee and Park, Jaesik},
  journal = {arXiv preprint arXiv:2605.30161},
  year    = {2026},
}
```
