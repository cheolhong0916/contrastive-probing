# Reference environments

Frozen package lists from the machine where `probing.py` was last verified
end-to-end against the paper's Table 5 (2026-05-25).

These are **reference snapshots, not strict requirements.** They reproduce
the exact verified setup; pinning every line is not necessary unless you hit
an incompatibility (see notes below).

## Two envs, why

| File                    | Used for                | Notable pins        |
|-------------------------|-------------------------|---------------------|
| `requirements-qwen.txt` / `conda-qwen.yaml`   | Qwen2.5-VL (qwen25), Qwen3-VL (qwen3) | transformers 4.57.6, torch 2.10 |
| `requirements-vila.txt` / `conda-vila.yaml`   | Molmo (molmo), NVILA-Lite (nvila)     | transformers 4.46.0, torch 2.3  |

Why two envs:

- **Molmo's bundled `modeling_molmo.py`** uses the legacy `past_key_values`
  tuple interface that was removed in `transformers >= 4.42` (replaced by
  `DynamicCache`). It crashes with `'NoneType' object has no attribute
  'size'` on every forward pass under newer transformers. Pin
  `transformers <= 4.46` for Molmo.
- **NVILA-Lite's custom `llava_llama` architecture** is registered by the
  VILA codebase and is happy on the same `transformers 4.46` line.
- **Qwen2.5-VL / Qwen3-VL** use the modern HF architecture and want
  `transformers >= 4.57` for the `Qwen2_5_VLModel.language_model.layers`
  layout. (The extractor falls back to the old path via `_find_llm_layers`,
  so older transformers also work for Qwen.)

## Quick reproduce

```bash
# Env A — for qwen25 / qwen3
conda env create -f envs/conda-qwen.yaml -n probing-qwen
# or, pip-only:
python -m venv .venv-qwen && source .venv-qwen/bin/activate
pip install -r envs/requirements-qwen.txt

# Env B — for molmo / nvila
conda env create -f envs/conda-vila.yaml -n probing-vila
# or, pip-only:
python -m venv .venv-vila && source .venv-vila/bin/activate
pip install -r envs/requirements-vila.txt
```

NVILA additionally requires the VILA repo on `PYTHONPATH` (see
[README "Installation"](../README.md#installation)).

## Notes

- These freezes include local-only paths (e.g. `file://...wheels/...`); strip
  or replace those for portable installs.
- CUDA driver / NCCL versions are baked into the torch wheel; if your
  driver is older than the one these wheels were built for, install a
  matching `torch` first, then `pip install -r ...`.
- The shell wrappers under `scripts/` leave `PYTHON=...` commented out —
  point them at whichever interpreter satisfies the env for each family.
