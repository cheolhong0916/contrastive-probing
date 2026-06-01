#!/usr/bin/env python3
"""
probing.py — Contrastive Probing for Spatial Representations
=============================================================================

Creates minimal contrastive pairs by swapping obj1 <-> obj2 in spatial questions:
  Original : "Is A to the left or right of B?"  → left
  Swapped  : "Is B to the left or right of A?"  → right

Spatial categories
------------------
  horizontal : left / right
  vertical   : above / below
  distance   : far / close

Analyses produced
-----------------
  1. Delta vectors         : Δr = h(r+) − h(r−) per layer
  2. Axis coherence        : mean pairwise cosine similarity of sign-corrected deltas
  3. VD-EI (from CSV)      : vertical–distance entanglement index via 6×6 matrix
  4. Delta similarity heatmap : 6×6 cosine similarity of mean Δ per category
  5. PCA visualisations    : 2-D and 3-D PCA of embeddings / delta vectors

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # Run a single registered model
  python probing.py --model_type qwen25 --scales vanilla

  # Run all checkpoints for a model family
  python probing.py --model_type qwen25

  # Generate cross-scale comparison plots (after inference)
  python probing.py --model_type qwen25 --merge

To add a new model, see the @register_model decorator and the MODEL_REGISTRY
section below.
"""

import os
import sys
import json
import argparse
import base64
import logging
import random
import re
from dataclasses import dataclass, field
from io import BytesIO
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Type, Any
from abc import ABC, abstractmethod

import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))

# ── Optional: set HF_HUB_DIR to a local mirror of the HuggingFace cache ─────
# If unset, models are downloaded from HuggingFace Hub as usual.
HF_HUB_DIR = os.environ.get('HF_HUB_DIR', '')


def resolve_model_path(model_id: str) -> str:
    """Return a local snapshot path if the model is cached, else return model_id unchanged."""
    if os.path.isabs(model_id):
        return model_id
    if HF_HUB_DIR:
        cache_name    = 'models--' + model_id.replace('/', '--')
        snapshots_dir = os.path.join(HF_HUB_DIR, cache_name, 'snapshots')
        if os.path.isdir(snapshots_dir):
            snapshots = sorted(os.listdir(snapshots_dir))
            if snapshots:
                local_path = os.path.join(snapshots_dir, snapshots[-1])
                # Only use local path if model weights are actually present.
                # An HF snapshot may exist with only config/tokenizer files
                # but no weights (partial download).
                weight_exts = ('.safetensors', '.bin', '.pt', '.pth')
                has_weights = any(
                    f.endswith(weight_exts)
                    for f in os.listdir(local_path)
                )
                if has_weights:
                    logger.info(f"Local cache found: {model_id}  →  {local_path}")
                    return local_path
                else:
                    logger.info(
                        f"Local snapshot for {model_id} has no weights "
                        f"(config-only cache). Falling back to HF download."
                    )
    return model_id


def _setup_file_logging(name: str, log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'{name}.log')
    fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(fh)
    return log_path


# ============================================================================
# Spatial Constants
# ============================================================================

CATEGORY_ORDER = ['left', 'right', 'above', 'below', 'far', 'close']

OPPOSITE_MAP = {
    'left': 'right', 'right': 'left',
    'above': 'below', 'below': 'above',
    'under': 'above',
    'far': 'close', 'close': 'far',
}

GROUP_MAP = {
    'left': 'horizontal', 'right': 'horizontal',
    'above': 'vertical',  'below': 'vertical',
    'far': 'distance',    'close': 'distance',
}

GROUP_ORDER = ['horizontal', 'vertical', 'distance']

CANONICAL_CATEGORIES = {
    'horizontal': 'left',
    'vertical':   'above',
    'distance':   'far',
}

# ── Question templates ────────────────────────────────────────────────────────

SHORT_TEMPLATES = {
    'horizontal': "Is the {obj1} to the left or right of the {obj2}? Answer with only one word.",
    'vertical':   "Is the {obj1} above or below the {obj2}? Answer with only one word.",
    'distance':   "Compared to {ref}, is {subj} far or close from you? Answer with only one word.",
}

# ── Plot style ────────────────────────────────────────────────────────────────

# Default colour for any scale not explicitly listed here.
_DEFAULT_SCALE_COLOR = '#7f7f7f'

SCALE_COLORS: Dict[str, str] = {
    'vanilla': '#1f77b4', '80k': '#ff7f0e', '400k': '#2ca02c',
    '800k': '#d62728', '2m': '#9467bd', 'roborefer': '#8c564b',
}

SCALE_ORDER: List[str] = [
    'vanilla', '80k', '400k', '800k', '2m', 'roborefer',
]

SCALE_DISPLAY_NAMES: Dict[str, str] = {}

CAT_COLORS = {
    'left':  '#ff7f0e', 'right':  '#ffbb78',
    'above': '#2ca02c', 'below':  '#98df8a',
    'far':   '#9467bd', 'close':  '#c5b0d5',
}

GROUP_COLORS = {
    'horizontal': '#ff7f0e',
    'vertical':   '#2ca02c',
    'distance':   '#9467bd',
}


# ============================================================================
# Model Registry
# ============================================================================

@dataclass
class ModelSpec:
    """Specification for a registered VLM.

    Attributes
    ----------
    extractor_class : Type[BaseHiddenStateExtractor]
        The extractor class that handles model loading and inference.
    checkpoints : dict
        Mapping from scale-name to HuggingFace model ID or absolute local path.
        E.g. {"base": "org/model-id", "ft": "/path/to/checkpoint"}.
    display_name : str
        Human-readable name shown in plot titles.
    base_processor_id : str
        When a fine-tuned checkpoint lacks its own tokenizer/processor config,
        set this to the HF ID of the base model to load the processor from there.
        Leave empty if each checkpoint is self-contained.
    plot_color : str
        Hex colour string used for this model in cross-model comparison plots.
    paper_layer : int, optional
        The single layer index used for the model's headline Coh_H / Coh_V /
        Coh_D / VD-EI numbers in the paper (Appendix D.1). Logged on extractor
        load so downstream scripts don't need to re-derive it. None means we
        haven't run the layer-selection sweep for this model.
    paper_plateau : (int, int), optional
        Inclusive (low, high) layer range over which coherence plateaus, per
        Appendix D.1. ``paper_layer`` is typically inside this band.
    total_layers : int, optional
        Total number of transformer layers in the LM backbone — useful so
        readers can see the relative depth (paper_layer / total_layers ≈ 71–93%
        across released models).
    """
    extractor_class: Type[Any]
    checkpoints: Dict[str, str]
    display_name: str = ''
    base_processor_id: str = ''
    plot_color: str = _DEFAULT_SCALE_COLOR
    paper_layer: Optional[int] = None
    paper_plateau: Optional[Tuple[int, int]] = None
    total_layers: Optional[int] = None


# ── Central registry — add your model here ────────────────────────────────────
#
# MODEL_REGISTRY maps  model_type_name  →  ModelSpec
#
# Built-in entries are populated at the bottom of the "Model Extractors" section.
# Users can add new entries in two ways:
#
#   (a) Directly, anywhere after defining the extractor class:
#         MODEL_REGISTRY["my_model"] = ModelSpec(
#             extractor_class = MyExtractor,
#             checkpoints     = {"v1": "org/my-model"},
#         )
#
#   (b) Via the @register_model decorator (see below).
#
MODEL_REGISTRY: Dict[str, ModelSpec] = {}


def register_model(
    model_type: str,
    checkpoints: Dict[str, str],
    display_name: str = '',
    base_processor_id: str = '',
    plot_color: str = _DEFAULT_SCALE_COLOR,
    paper_layer: Optional[int] = None,
    paper_plateau: Optional[Tuple[int, int]] = None,
    total_layers: Optional[int] = None,
):
    """Class decorator that registers a VLM extractor in MODEL_REGISTRY.

    Example
    -------
    @register_model(
        model_type   = "my_model",
        checkpoints  = {"base": "org/my-model-7b"},
        display_name = "My Model 7B",
        plot_color   = "#e377c2",
        paper_layer  = 23,           # see Appendix D.1 — selected analysis layer
        paper_plateau= (20, 25),     # plateau range coherence is stable over
        total_layers = 32,
    )
    class MyModelExtractor(BaseHiddenStateExtractor):
        ...
    """
    def decorator(cls):
        MODEL_REGISTRY[model_type] = ModelSpec(
            extractor_class  = cls,
            checkpoints      = checkpoints,
            display_name     = display_name or model_type,
            base_processor_id= base_processor_id,
            plot_color       = plot_color,
            paper_layer      = paper_layer,
            paper_plateau    = paper_plateau,
            total_layers     = total_layers,
        )
        return cls
    return decorator


# ── Merge / compare configs (cross-model comparisons) ─────────────────────────
#
# A merge config combines results from several model_types into one comparison plot.
# Keys in scale_sources must match registered model_type names.
#
MERGE_CONFIGS: Dict[str, dict] = {
    # Example:
    # "qwen_all": {
    #     "scale_order":   ["3b", "32b"],
    #     "scale_sources": {"3b": "qwen25", "32b": "qwen3"},
    # },
}


# ============================================================================
# Base Extractor
# ============================================================================

class BaseHiddenStateExtractor(ABC):
    """Abstract base class for VLM hidden-state extractors.

    Subclass this and implement the four abstract methods to support a new model.
    Hook registration and cleanup are handled automatically by this base class.
    """

    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        target_layers: Optional[List[int]] = None,
        base_processor_id: str = '',
    ):
        self.model_path = model_path
        self.device = device
        self.base_processor_id = base_processor_id
        self.hidden_states: Dict[int, torch.Tensor] = {}
        self.hooks: List = []

        self._load_model()
        num_layers = self._get_num_layers()

        if target_layers is None:
            self.target_layers = list(range(num_layers))
            logger.info(f"Model has {num_layers} layers — extracting ALL.")
        else:
            self.target_layers = target_layers

        self._register_hooks()

    # ── Hook infrastructure ───────────────────────────────────────────────────

    def _register_hooks(self):
        for layer_idx in self.target_layers:
            module = self._get_layer_module(layer_idx)
            if module is not None:
                hook = module.register_forward_hook(self._make_hook(layer_idx))
                self.hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.shape[1] > 1:  # prefill only (not single-token decode)
                self.hidden_states[layer_idx] = hidden[:, -1, :].detach().cpu().float().squeeze(0)
        return hook_fn

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def _load_model(self):
        """Load self.model (and self.processor if needed)."""

    @abstractmethod
    def _get_num_layers(self) -> int:
        """Return the total number of transformer layers."""

    @abstractmethod
    def _get_layer_module(self, layer_idx: int):
        """Return the nn.Module for transformer layer *layer_idx*."""

    @abstractmethod
    def extract_and_predict(
        self, image: Image.Image, question: str
    ) -> Tuple[Dict[int, torch.Tensor], str]:
        """Run inference and return (hidden_states_dict, answer_string).

        Must set ``self.hidden_states = {}`` at the start so that each call
        returns a fresh snapshot captured by the hooks.
        """

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def cleanup(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        for attr in ('model', 'processor', 'tokenizer'):
            if hasattr(self, attr):
                delattr(self, attr)
        torch.cuda.empty_cache()

    # ── Shared helper: scan all named modules for transformer layers ───────────

    def _find_layers_by_scan(self, hint: str = '') -> Any:
        """Return the largest ModuleList named '*.layers' found by scanning the model.

        Useful as a fallback when the exact attribute path is unknown.
        """
        best, best_len = None, 0
        for name, module in self.model.named_modules():
            if name.endswith('.layers') and hasattr(module, '__len__') and len(module) > best_len:
                best, best_len = module, len(module)
                logger.info(f"{hint or type(self).__name__}: layers via scan at '{name}', n={best_len}")
        return best


# ============================================================================
# Model Extractors
# ============================================================================
# Each extractor handles one model family.
# Use @register_model to both define and register the extractor in one step,
# or register manually via MODEL_REGISTRY["name"] = ModelSpec(...) after the class.

# ── Helper shared by Molmo2 / Qwen3 ─────────────────────────────────────────────────

def _find_llm_layers(model, candidate_paths: List[List[str]], hint: str = ''):
    """Try a list of attribute paths in order; return the first valid layers list."""
    for path in candidate_paths:
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, '__len__') and len(obj) > 0:
            logger.info(f"{hint}: layers at '{'.'.join(path)}', n={len(obj)}")
            return obj
    # Fallback: scan — prefer language-model decoder layers over vision encoder layers.
    # Rank by: (1) contains Decoder/Attention layer types, (2) then by length.
    LM_LAYER_KEYWORDS = ('Decoder', 'Attention', 'Transformer')
    VIS_LAYER_KEYWORDS = ('Vision', 'Siglip', 'Clip', 'Vit')
    best, best_score = None, (-1, 0)
    for name, module in model.named_modules():
        if not (name.endswith('.layers') and hasattr(module, '__len__') and len(module) > 0):
            continue
        n = len(module)
        # Check the type name of contained layers
        child_cls = type(next(iter(module))).__name__ if n > 0 else ''
        is_lm  = any(k in child_cls for k in LM_LAYER_KEYWORDS)
        is_vis = any(k in child_cls for k in VIS_LAYER_KEYWORDS)
        score = (1 if is_lm else (-1 if is_vis else 0), n)
        if score > best_score:
            best, best_score = module, score
            logger.info(f"{hint}: layers via scan at '{name}', n={n}, cls={child_cls}")
    return best


# ── Qwen2.5-VL ───────────────────────────────────────────────────────────────

class Qwen25VLExtractor(BaseHiddenStateExtractor):
    """Extractor for Qwen/Qwen2.5-VL family."""

    def _load_model(self):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16, device_map=self.device
        ).eval()
        proc_id = self.base_processor_id or self.model_path
        self.processor = AutoProcessor.from_pretrained(proc_id)
        self.llm_layers = _find_llm_layers(self.model, [
            ['model', 'layers'],
            ['model', 'language_model', 'layers'],
            ['model', 'language_model', 'model', 'layers'],
        ], hint='Qwen2.5-VL')
        if self.llm_layers is None:
            raise ValueError("Could not find transformer layers in Qwen2.5-VL model")
        logger.info(f"Loaded Qwen2.5-VL from {self.model_path}")

    def _get_num_layers(self) -> int:
        return len(self.llm_layers)

    def _get_layer_module(self, layer_idx: int):
        return self.llm_layers[layer_idx]

    def extract_and_predict(self, image, question):
        self.hidden_states = {}
        from qwen_vl_utils import process_vision_info
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text":  question},
        ]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=20, do_sample=False)
        input_len = inputs['input_ids'].shape[1]
        answer = self.processor.tokenizer.decode(output_ids[0, input_len:], skip_special_tokens=True).strip()
        return self.hidden_states.copy(), answer


# ── Qwen3-VL ─────────────────────────────────────────────────────────────────

class Qwen3VLExtractor(BaseHiddenStateExtractor):
    """Extractor for Qwen3-VL family (32B dense, 235B MoE)."""

    MIN_PIXELS = 256   * 32 * 32
    MAX_PIXELS = 16384 * 32 * 32

    def _load_model(self):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path, trust_remote_code=True, torch_dtype='auto',
            device_map='auto', attn_implementation='flash_attention_2',
        ).eval()
        self.llm_layers = _find_llm_layers(self.model, [
            ['model', 'language_model', 'model', 'layers'],
            ['language_model', 'model', 'layers'],
            ['model', 'model', 'layers'],
            ['model', 'layers'],
        ], hint='Qwen3-VL')
        if self.llm_layers is None:
            raise ValueError("Could not find transformer layers in Qwen3-VL model")
        logger.info(f"Loaded Qwen3-VL from {self.model_path}")

    def _get_num_layers(self) -> int:
        return len(self.llm_layers)

    def _get_layer_module(self, layer_idx: int):
        return self.llm_layers[layer_idx]

    def extract_and_predict(self, image, question):
        self.hidden_states = {}
        from qwen_vl_utils import process_vision_info
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image,
             "min_pixels": self.MIN_PIXELS, "max_pixels": self.MAX_PIXELS},
            {"type": "text", "text": question},
        ]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, _ = process_vision_info(
            messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True,
        )
        inputs = self.processor(
            text=text, images=images, videos=videos, do_resize=False, return_tensors="pt",
        ).to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=20, do_sample=False)
        input_len = inputs['input_ids'].shape[1]
        answer = self.processor.tokenizer.decode(
            output_ids[0, input_len:], skip_special_tokens=True).strip()
        return self.hidden_states.copy(), answer


# ── Molmo 7B (Allen AI) ───────────────────────────────────────────────────────

class MolmoExtractor(BaseHiddenStateExtractor):
    """Extractor for allenai/Molmo-7B-* (HuggingFace variant)."""

    def _load_model(self):
        from transformers import AutoModelForCausalLM, AutoProcessor
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True, device_map=self.device,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        logger.info(f"Loaded Molmo (HF) from {self.model_path}")

    def _get_num_layers(self) -> int:
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'transformer'):
            return len(self.model.model.transformer.blocks)
        return 32

    def _get_layer_module(self, layer_idx: int):
        return self.model.model.transformer.blocks[layer_idx]

    def extract_and_predict(self, image, question):
        self.hidden_states = {}
        from transformers import GenerationConfig
        inputs = self.processor.process(images=[image], text=question)
        processed = {k: v.to(self.device).unsqueeze(0) for k, v in inputs.items()}
        for k in processed:
            if processed[k].dtype == torch.float32:
                processed[k] = processed[k].to(dtype=torch.bfloat16)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.model.generate_from_batch(
                processed,
                GenerationConfig(max_new_tokens=20, stop_strings="<|endoftext|>"),
                tokenizer=self.processor.tokenizer,
            )
        input_len = processed['input_ids'].shape[1]
        answer = self.processor.tokenizer.decode(output[0, input_len:], skip_special_tokens=True).strip()
        return self.hidden_states.copy(), answer


# ── Molmo2 8B ─────────────────────────────────────────────────────────────────

class Molmo2Extractor(BaseHiddenStateExtractor):
    """Extractor for allenai/Molmo2-8B (AutoModelForImageTextToText)."""

    def _load_model(self):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path, trust_remote_code=True, torch_dtype='auto', device_map='auto',
        ).eval()
        self.llm_layers = _find_llm_layers(self.model, [
            ['model', 'layers'],
            ['language_model', 'model', 'layers'],
            ['model', 'model', 'layers'],
        ], hint='Molmo2')
        if self.llm_layers is None:
            raise ValueError("Could not find transformer layers in Molmo2 model")
        logger.info(f"Loaded Molmo2 from {self.model_path}")

    def _get_num_layers(self) -> int:
        return len(self.llm_layers)

    def _get_layer_module(self, layer_idx: int):
        return self.llm_layers[layer_idx]

    def extract_and_predict(self, image, question):
        self.hidden_states = {}
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text":  question},
        ]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=20, do_sample=False)
        input_len = inputs['input_ids'].shape[1]
        answer = self.processor.tokenizer.decode(
            generated_ids[0, input_len:], skip_special_tokens=True).strip()
        return self.hidden_states.copy(), answer


# ── NVILA ─────────────────────────────────────────────────────────────────────

class NVILAExtractor(BaseHiddenStateExtractor):
    """Extractor for NVILA / NVILA-Lite family (uses llava package)."""

    LLAVA_PACKAGE_PATH: str = ''   # Override if llava is not on sys.path

    def _load_model(self):
        original_sys_path = sys.path.copy()
        modules_to_remove = [k for k in list(sys.modules.keys()) if 'llava' in k.lower()]
        removed = {m: sys.modules.pop(m) for m in modules_to_remove}
        try:
            import llava
            from llava.media import Image as LLaVAImage
            from llava import conversation as clib
        except Exception as err:
            sys.path = original_sys_path
            for m, mod in removed.items():
                sys.modules[m] = mod
            raise RuntimeError(f"Failed to import llava: {err}") from err
        sys.path = original_sys_path
        self.LLaVAImage = LLaVAImage
        self.clib = clib
        self.model = llava.load(self.model_path, model_base=None)
        self._find_llm_backbone()
        logger.info(f"Loaded NVILA from {self.model_path}")

    def _find_llm_backbone(self):
        for name, module in self.model.named_modules():
            if name.endswith('.layers') and hasattr(module, '__len__') and len(module) > 0:
                self.llm_backbone = module
                return
        raise ValueError("Could not locate transformer layers in NVILA model")

    def _get_num_layers(self) -> int:
        return len(self.llm_backbone) if hasattr(self, 'llm_backbone') else 24

    def _get_layer_module(self, layer_idx: int):
        return self.llm_backbone[layer_idx]

    def extract_and_predict(self, image, question):
        self.hidden_states = {}
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
            image.save(temp_path)
        try:
            from transformers import GenerationConfig
            response = self.model.generate_content(
                [self.LLaVAImage(temp_path), question],
                generation_config=GenerationConfig(max_new_tokens=20, do_sample=False),
            )
        finally:
            os.unlink(temp_path)
        answer = str(response[0] if isinstance(response, list) else response).strip()
        return self.hidden_states.copy(), answer



# ============================================================================
# MODEL_REGISTRY — built-in entries
# ============================================================================
# Update the "checkpoints" paths to point to your local fine-tuned checkpoints.
# HuggingFace model IDs are used as defaults and will be downloaded automatically
# if not cached locally.

MODEL_REGISTRY["qwen25"] = ModelSpec(
    extractor_class   = Qwen25VLExtractor,
    checkpoints       = {
        "vanilla" : "Qwen/Qwen2.5-VL-3B-Instruct",
        "80k"     : "ch-min/Qwen2.5-VL-3B-Instruct-data_scale_exp_80k-20251114_120221",
        "400k"    : "ch-min/Qwen2.5-VL-3B-Instruct-data_scale_exp_400k-20251114_120221",
        "800k"    : "ch-min/Qwen2.5-VL-3B-Instruct-data_scale_exp_800k-20251114_120221",
        "2m"      : "ch-min/Qwen2.5-VL-3B-Instruct-data_scale_exp_2m-20260109_120517",
    },
    display_name      = "Qwen2.5-VL",
    base_processor_id = "Qwen/Qwen2.5-VL-3B-Instruct",
    plot_color        = "#1f77b4",
    # Appendix D.1: L*=28 (78% of 36); plateau L20–28.
    paper_layer       = 28,
    paper_plateau     = (20, 28),
    total_layers      = 36,
)

MODEL_REGISTRY["qwen3"] = ModelSpec(
    extractor_class   = Qwen3VLExtractor,
    checkpoints       = {
        "32b"  : "Qwen/Qwen3-VL-32B-Instruct",
        "235b" : "Qwen/Qwen3-VL-235B-A22B-Instruct",
    },
    display_name = "Qwen3-VL",
    plot_color   = "#bcbd22",
    # Appendix D.1 reports L*=87 (93% of 94) for the 235B MoE variant; 32B not
    # included in the layer-selection sweep. VD-EI oscillates in the upper
    # layers without a clear plateau (see paper caveat).
    paper_layer  = 87,
    paper_plateau= (83, 90),
    total_layers = 94,
)

MODEL_REGISTRY["molmo"] = ModelSpec(
    extractor_class = MolmoExtractor,
    checkpoints     = {
        "vanilla" : "allenai/Molmo-7B-O-0924",
    },
    display_name = "Molmo-7B",
    plot_color   = "#ff7f0e",
    # Appendix D.1: L*=23 (72% of 32); plateau L20–25.
    paper_layer  = 23,
    paper_plateau= (20, 25),
    total_layers = 32,
)

MODEL_REGISTRY["molmo2"] = ModelSpec(
    extractor_class = Molmo2Extractor,
    checkpoints     = {
        "8b" : "allenai/Molmo2-8B",
    },
    display_name = "Molmo2-8B",
    plot_color   = "#17becf",
    # Not included in Appendix D.1 layer-selection sweep — pick after running.
)

MODEL_REGISTRY["nvila"] = ModelSpec(
    extractor_class = NVILAExtractor,
    checkpoints     = {
        "vanilla"   : "Efficient-Large-Model/NVILA-Lite-2B",
        "80k"       : "ch-min/NVILA-Lite-2B-DATA_SCALE_EXP_80K-20251108_180221",
        "400k"      : "ch-min/NVILA-Lite-2B-DATA_SCALE_EXP_400K-20251108_180221",
        "800k"      : "ch-min/NVILA-Lite-2B-DATA_SCALE_EXP_800K-20251108_180221",
        "2m"        : "ch-min/NVILA-Lite-2B-DATA_SCALE_EXP_2M-20260205_003632",
        "roborefer" : "Zhoues/RoboRefer-2B-SFT",  # third-party checkpoint
    },
    display_name = "NVILA-Lite",
    plot_color   = "#2ca02c",
    # Appendix D.1: L*=20 (71% of 28); plateau L17–26.
    paper_layer  = 20,
    paper_plateau= (17, 26),
    total_layers = 28,
)

# ── Example merge config ──────────────────────────────────────────────────────
# Uncomment and customise to compare Qwen2.5 and Qwen3 side by side:
#
# MERGE_CONFIGS["qwen_all"] = {
#     "scale_order"  : ["3b", "32b"],
#     "scale_sources": {"3b": "qwen25", "32b": "qwen3"},
# }


def get_extractor(
    model_type: str,
    scale: str,
    device: str = 'cuda',
    target_layers: Optional[List[int]] = None,
) -> BaseHiddenStateExtractor:
    """Instantiate the extractor for *model_type* at *scale*."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    spec = MODEL_REGISTRY[model_type]
    if scale not in spec.checkpoints:
        raise ValueError(
            f"Scale '{scale}' not in MODEL_REGISTRY['{model_type}'].checkpoints. "
            f"Available scales: {list(spec.checkpoints.keys())}"
        )
    raw_path = spec.checkpoints[scale]
    model_path = resolve_model_path(raw_path)
    logger.info(f"Loading {model_type}/{scale} via {type(spec.extractor_class).__name__} "
                f"from {model_path}")
    return spec.extractor_class(
        model_path        = model_path,
        device            = device,
        target_layers     = target_layers,
        base_processor_id = resolve_model_path(spec.base_processor_id) if spec.base_processor_id else '',
    )


def _scale_color(model_type: str, scale: str) -> str:
    """Return a colour for this (model_type, scale) combination."""
    key = f"{model_type}_{scale}"
    if key in SCALE_COLORS:
        return SCALE_COLORS[key]
    if model_type in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_type].plot_color
    return _DEFAULT_SCALE_COLOR


def _scale_display(model_type: str, scale: str) -> str:
    key = f"{model_type}_{scale}"
    return SCALE_DISPLAY_NAMES.get(key, SCALE_DISPLAY_NAMES.get(scale, scale))


# ============================================================================
# Data Loading
# ============================================================================

OBJECT_PATTERNS = [
    re.compile(r'between\s+(.+?)\s+and\s+(.+?)\s+in',       re.IGNORECASE),
    re.compile(r'of\s+(.+?)\s+and\s+(.+?)\s+in',            re.IGNORECASE),
    re.compile(r'positions\s+of\s+(.+?)\s+and\s+(.+?)\s+interact', re.IGNORECASE),
    re.compile(r'How\s+are\s+(.+?)\s+and\s+(.+?)\s+positioned',    re.IGNORECASE),
    re.compile(r'arrangement\s+of\s+(.+?)\s+and\s+(.+?)\s+in',     re.IGNORECASE),
]


def extract_objects(question: str) -> Tuple[str, str]:
    for pattern in OBJECT_PATTERNS:
        m = pattern.search(question)
        if m:
            return m.group(1).strip(), m.group(2).strip()
    raise ValueError(f"Could not extract objects from: {question}")


def decode_base64_image(base64_str: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(base64_str))).convert('RGB')


# ── Swap pair creation ────────────────────────────────────────────────────────

def load_swap_pairs(
    tsv_path: str,
    seed: int = 42,
) -> List[dict]:
    """Load EmbSpatial-Bench TSV and create minimal contrastive swap pairs.

    For far/close (distance) pairs, samples with unknown/empty target or
    reference objects are filtered out.
    """
    rng = random.Random(seed)
    df  = pd.read_csv(tsv_path, sep='\t')

    pairs = []
    stats = defaultdict(lambda: {'total': 0, 'success': 0})

    def _valid_obj(v):
        return bool(v) and str(v).strip().lower() not in ('unknown', 'n/a', '')

    for _, row in df.iterrows():
        category = row['category']
        # Normalise dataset aliases to canonical category names before any stats/logic
        if category in ('under', 'beneath'):
            category = 'below'
        stats[category]['total'] += 1
        try:
            if category in ['left', 'right', 'above', 'below']:
                obj1, obj2 = extract_objects(row['question'])
                grp = 'horizontal' if category in ['left', 'right'] else 'vertical'
                tmpl = SHORT_TEMPLATES[grp]
                pair = {
                    'index': row['index'], 'question_id': str(row['question_id']),
                    'image_base64': row['image'],
                    'original_question': tmpl.format(obj1=obj1, obj2=obj2),
                    'swapped_question':  tmpl.format(obj1=obj2, obj2=obj1),
                    'original_answer': category,
                    'swapped_answer': OPPOSITE_MAP[category],
                    'group': grp, 'category': category,
                    'obj1': obj1, 'obj2': obj2,
                }

            elif category in ['far', 'close']:
                answer_key     = row['answer']
                options        = {k: row[k] for k in ['A', 'B', 'C', 'D']}
                target_object  = options[answer_key]
                candidates     = [v for k, v in options.items() if k != answer_key]

                if not _valid_obj(target_object):
                    continue
                candidates = [v for v in candidates if _valid_obj(v)]
                if not candidates:
                    continue

                reference_object = rng.choice(candidates)
                tmpl = SHORT_TEMPLATES['distance']
                pair = {
                    'index': row['index'], 'question_id': str(row['question_id']),
                    'image_base64': row['image'],
                    'original_question': tmpl.format(ref=reference_object, subj=target_object),
                    'swapped_question':  tmpl.format(ref=target_object,  subj=reference_object),
                    'original_answer': category,
                    'swapped_answer': OPPOSITE_MAP[category],
                    'group': 'distance', 'category': category,
                    'target_object': target_object, 'reference_object': reference_object,
                }
            else:
                continue

            pairs.append(pair)
            stats[category]['success'] += 1

        except Exception as e:
            logger.warning(f"Failed to create swap pair for index {row['index']}: {e}")

    logger.info("Swap pair creation stats:")
    for cat in CATEGORY_ORDER:
        s = stats[cat]
        logger.info(f"  {cat}: {s['success']}/{s['total']}")
    logger.info(f"  Total: {len(pairs)}")
    return pairs


# ============================================================================
# Feature Extraction Pipeline
# ============================================================================

def _run_query(extractor: BaseHiddenStateExtractor, image: Image.Image, question: str):
    hidden_states, predicted = extractor.extract_and_predict(image, question)
    result = {}
    for layer_idx in extractor.target_layers:
        if layer_idx in hidden_states:
            state = hidden_states[layer_idx].numpy().flatten()
            if state.size > 0:
                result[layer_idx] = state
    return result, predicted


def extract_swap_features(
    extractor: BaseHiddenStateExtractor,
    swap_pairs: List[dict],
    max_samples_per_category: int = 0,
) -> List[dict]:
    """Extract hidden states for all swap pairs."""
    rng = random.Random(42)
    if max_samples_per_category > 0:
        grouped = defaultdict(list)
        for p in swap_pairs:
            grouped[p['category']].append(p)
        swap_pairs = []
        for cat in CATEGORY_ORDER:
            samples = grouped[cat]
            if len(samples) > max_samples_per_category:
                samples = rng.sample(samples, max_samples_per_category)
            swap_pairs.extend(samples)

    records = []
    for pair in tqdm(swap_pairs, desc="Swap pairs"):
        try:
            image      = decode_base64_image(pair['image_base64'])
            hs_o, p_o  = _run_query(extractor, image, pair['original_question'])
            hs_s, p_s  = _run_query(extractor, image, pair['swapped_question'])

            delta = {
                l: hs_s[l] - hs_o[l]
                for l in extractor.target_layers
                if l in hs_o and l in hs_s
            }

            records.append({
                'index': pair['index'], 'group': pair['group'], 'category': pair['category'],
                'original_answer': pair['original_answer'], 'swapped_answer': pair['swapped_answer'],
                'pred_orig': p_o, 'pred_swap': p_s,
                'hs_orig': hs_o, 'hs_swap': hs_s, 'delta': delta,
            })

            logger.info(f"  #{pair['index']:<6} {pair['category']:<6} "
                        f"orig=\"{p_o[:40]}\" swap=\"{p_s[:40]}\"")

        except Exception as e:
            logger.warning(f"Error on index {pair['index']}: {e}")

    logger.info(f"Extracted {len(records)} records")
    for cat in CATEGORY_ORDER:
        n = sum(1 for r in records if r['category'] == cat)
        if n:
            logger.info(f"  {cat:>6s}: n={n}")
    return records


# ============================================================================
# Analysis Functions
# ============================================================================

def compute_axis_coherence(records: List[dict], target_layers: List[int]) -> dict:
    """Axis coherence (Eq. 2): mean pairwise cosine similarity of sign-corrected delta vectors per group."""
    results = {}
    for group in GROUP_ORDER:
        canonical = CANONICAL_CATEGORIES[group]
        opposite  = OPPOSITE_MAP[canonical]
        group_recs = [r for r in records if r['group'] == group]

        for layer in target_layers:
            all_deltas = []
            for r in group_recs:
                if layer not in r['delta']:
                    continue
                d = r['delta'][layer]
                if r['category'] == opposite:
                    d = -d
                all_deltas.append(d)

            if len(all_deltas) >= 2:
                arr = np.array(all_deltas)
                sim = cosine_similarity(arr)
                upper = sim[np.triu_indices(len(all_deltas), k=1)]
                results[(group, layer)] = {
                    'mean': float(np.mean(upper)),
                    'std':  float(np.std(upper)),
                    'n':    len(all_deltas),
                }

    return results


def compute_delta_similarity_matrix(records: List[dict], layer: int) -> Optional[pd.DataFrame]:
    """6×6 cosine similarity matrix using mean delta per category."""
    cat_deltas = {}
    for cat in CATEGORY_ORDER:
        deltas = [r['delta'][layer] for r in records if r['category'] == cat and layer in r['delta']]
        if deltas:
            cat_deltas[cat] = np.mean(deltas, axis=0)
    available = [c for c in CATEGORY_ORDER if c in cat_deltas]
    if len(available) < 2:
        return None
    vectors = np.array([cat_deltas[c] for c in available])
    sim = cosine_similarity(vectors)
    return pd.DataFrame(sim, index=available, columns=available)


def compute_vd_ei_per_layer(records: List[dict], target_layers: List[int]) -> Dict[int, float]:
    """Per-layer VD-Entanglement Index.

    For each layer, compute cosine(mean_delta_vertical, mean_delta_distance)
    after sign-correcting each group's deltas onto its canonical direction
    (e.g. flip Δ for samples whose category is `below` so they align with the
    canonical `above` direction before averaging). Without this flip the group
    mean cancels to ~0 and the cosine becomes meaningless noise.

    Reference implementation: rb/R3_M3_refspatial_rgb_finetune/watch_400k.sh
        d[c == opp] = -d[c == opp]      # sign-correct
        means[grp]  = d.mean(axis=0)
        vdei        = dot(mv, md) / (||mv|| * ||md||)

    Returns a {layer_idx: vdei} dict; layers missing either group are skipped.
    """
    out: Dict[int, float] = {}
    for layer in target_layers:
        means = {}
        for group in ('vertical', 'distance'):
            canon = CANONICAL_CATEGORIES[group]
            opp   = OPPOSITE_MAP[canon]
            deltas = []
            for r in records:
                if r['group'] != group or layer not in r['delta']:
                    continue
                d = r['delta'][layer]
                if r['category'] == opp:
                    d = -d
                deltas.append(d)
            if deltas:
                means[group] = np.mean(deltas, axis=0)
        if 'vertical' in means and 'distance' in means:
            v, d = means['vertical'], means['distance']
            denom = (np.linalg.norm(v) * np.linalg.norm(d) + 1e-12)
            out[layer] = float(np.dot(v, d) / denom)
    return out


def compute_vd_ei_per_layer_from_npz(npz_path: str) -> Dict[int, float]:
    """Same as ``compute_vd_ei_per_layer`` but reads delta vectors directly from
    a saved ``vectors_<scale>.npz``. Lets ``--rebuild-metrics`` regenerate plots
    from existing artifacts without redoing inference."""
    if not os.path.exists(npz_path):
        return {}
    data = np.load(npz_path, allow_pickle=True)
    layers = sorted({int(k.split('_L')[1]) for k in data.files if k.startswith('delta_L')})
    out: Dict[int, float] = {}
    for L in layers:
        dkey, ckey, gkey = f'delta_L{L}', f'categories_L{L}', f'groups_L{L}'
        if not (dkey in data.files and ckey in data.files and gkey in data.files):
            continue
        delta = data[dkey].astype(np.float32)
        cats  = data[ckey]
        grps  = data[gkey]
        means = {}
        for group in ('vertical', 'distance'):
            canon = CANONICAL_CATEGORIES[group]
            opp   = OPPOSITE_MAP[canon]
            mask  = (grps == group)
            d = delta[mask].copy()
            c = cats[mask]
            d[c == opp] = -d[c == opp]
            if len(d):
                means[group] = d.mean(axis=0)
        if 'vertical' in means and 'distance' in means:
            v, d = means['vertical'], means['distance']
            denom = (np.linalg.norm(v) * np.linalg.norm(d) + 1e-12)
            out[L] = float(np.dot(v, d) / denom)
    return out




# ============================================================================
# Saving & Loading
# ============================================================================

def save_scale_results(scale, swap_records, axis_coherence, delta_heatmaps, output_dir):
    csv_dir  = os.path.join(output_dir, 'csv')
    json_dir = os.path.join(output_dir, 'json')
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    coh_data = {f'{group}_L{layer}': vals for (group, layer), vals in axis_coherence.items()}
    with open(os.path.join(json_dir, f'axis_coherence_{scale}.json'), 'w') as f:
        json.dump(coh_data, f, indent=2)

    for layer, df in delta_heatmaps.items():
        if df is not None:
            df.to_csv(os.path.join(csv_dir, f'delta_similarity_{scale}_L{layer}.csv'))

    logger.info(f"Saved results for scale={scale} to {output_dir}")


def save_vectors_npz(scale, swap_records, target_layers, output_dir):
    delta_data = {}
    for layer in target_layers:
        groups_, cats_, vecs_, idxs_ = [], [], [], []
        orig_, swap_, lbls_ = [], [], []
        for r in swap_records:
            if layer in r['delta']:
                groups_.append(r['group']); cats_.append(r['category'])
                vecs_.append(r['delta'][layer])
                idxs_.append(r['index'])
            if layer in r['hs_orig'] and layer in r['hs_swap']:
                orig_.append(r['hs_orig'][layer]); swap_.append(r['hs_swap'][layer])
                lbls_.append(r['category'])
        if vecs_:
            delta_data[f'delta_L{layer}']      = np.array(vecs_)
            delta_data[f'groups_L{layer}']     = np.array(groups_)
            delta_data[f'categories_L{layer}'] = np.array(cats_)
            delta_data[f'indices_L{layer}']    = np.array(idxs_)
        if orig_:
            delta_data[f'orig_L{layer}']   = np.array(orig_)
            delta_data[f'swap_L{layer}']   = np.array(swap_)
            delta_data[f'labels_L{layer}'] = np.array(lbls_)

    npz_dir = os.path.join(output_dir, 'npz')
    os.makedirs(npz_dir, exist_ok=True)
    np.savez_compressed(os.path.join(npz_dir, f'vectors_{scale}.npz'), **delta_data)
    logger.info(f"Saved vectors NPZ for scale={scale}")


def load_axis_coherence(output_dir, scale):
    path = os.path.join(output_dir, 'json', f'axis_coherence_{scale}.json')
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = json.load(f)
    return {(p[0], int(p[1])): vals for key, vals in raw.items()
            for p in [key.rsplit('_L', 1)] if len(p) == 2}


def load_vd_ei(output_dir, scale):
    path = os.path.join(output_dir, 'json', f'vd_ei_{scale}.json')
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {int(k): float(v) for k, v in json.load(f).items()}


def save_vd_ei(output_dir, scale, vd_ei: Dict[int, float]):
    json_dir = os.path.join(output_dir, 'json')
    os.makedirs(json_dir, exist_ok=True)
    with open(os.path.join(json_dir, f'vd_ei_{scale}.json'), 'w') as f:
        json.dump({str(k): float(v) for k, v in sorted(vd_ei.items())}, f, indent=2)


def load_delta_heatmaps(output_dir, scale):
    import glob as glob_mod
    pattern = os.path.join(output_dir, 'csv', f'delta_similarity_{scale}_L*.csv')
    result  = {}
    for fpath in glob_mod.glob(pattern):
        m = re.search(
            rf'delta_similarity_{re.escape(scale)}_L(\d+)\.csv$',
            os.path.basename(fpath))
        if m:
            result[int(m.group(1))] = pd.read_csv(fpath, index_col=0)
    return result


def _model_key(model_type: str, scale: str) -> str:
    return f"{model_type}_{scale}"


# ============================================================================
# Visualization
# ============================================================================

def plot_axis_coherence_trajectory(axis_coherence, scale, model_type, save_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    for group in GROUP_ORDER:
        layers, vals = zip(*sorted(
            ((l, v['mean']) for (g, l), v in axis_coherence.items() if g == group),
            key=lambda x: x[0]
        )) if any(g == group for (g, _) in axis_coherence) else ([], [])
        if layers:
            ax.plot(layers, vals, '-o', color=GROUP_COLORS[group], label=group, linewidth=2, markersize=3)
    ax.set_xlabel('Layer Index'); ax.set_ylabel('Axis Coherence')
    ax.set_title(f'{model_type} ({scale}) – Axis Coherence', fontweight='bold')
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
    logger.info(f"Saved: {save_path}")


def plot_delta_heatmap(sim_df, title, save_path):
    plt.figure(figsize=(10, 8))
    ordered = [c for c in CATEGORY_ORDER if c in sim_df.index]
    df_o = sim_df.loc[ordered, ordered]
    sns.heatmap(df_o, annot=df_o.round(4).astype(str), fmt='', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
                cbar_kws={'label': 'Cosine Similarity'})
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
    logger.info(f"Saved: {save_path}")


def plot_cross_scale_axis_coherence(all_consistency, model_type, save_path, title_prefix='Axis Coherence'):
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    all_cohales = list(all_consistency.keys())
    for idx, group in enumerate(GROUP_ORDER):
        ax = axes[idx]
        for scale in all_cohales:
            pairs = sorted(
                ((l, v['mean']) for (g, l), v in all_consistency[scale].items() if g == group),
                key=lambda x: x[0])
            if pairs:
                ls, vs = zip(*pairs)
                ax.plot(ls, vs, '-', color=SCALE_COLORS.get(scale, _DEFAULT_SCALE_COLOR),
                        label=_scale_display('', scale), linewidth=2)
        ax.set_xlabel('Layer Index'); ax.set_ylabel('Consistency')
        ax.set_title(group, fontweight='bold'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.suptitle(f'{model_type} – {title_prefix} Consistency Across Scales',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
    logger.info(f"Saved: {save_path}")


def plot_vd_ei_trajectory(vd_ei: Dict[int, float], scale, model_type, save_path):
    """Per-layer VD-EI trajectory for one (model, scale)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    pairs = sorted(vd_ei.items())
    if pairs:
        layers, vals = zip(*pairs)
        ax.plot(layers, vals, '-o', color='#7f0000', linewidth=2, markersize=3, label='VD-EI')
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xlabel('Layer Index'); ax.set_ylabel('VD-EI')
    ax.set_title(f'{model_type} ({scale}) – VD-Entanglement Index', fontweight='bold')
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
    logger.info(f"Saved: {save_path}")


def plot_cross_scale_vd_ei(all_vd_ei, model_type, save_path):
    """VD-EI trajectories overlaid across scales for one model."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for scale, vd_ei in all_vd_ei.items():
        pairs = sorted(vd_ei.items())
        if pairs:
            ls, vs = zip(*pairs)
            ax.plot(ls, vs, '-', color=SCALE_COLORS.get(scale, _DEFAULT_SCALE_COLOR),
                    label=_scale_display('', scale), linewidth=2)
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xlabel('Layer Index'); ax.set_ylabel('VD-EI')
    ax.set_title(f'{model_type} – VD-Entanglement Index Across Scales',
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
    logger.info(f"Saved: {save_path}")


def rebuild_metrics_for_scale(scale_output_dir: str, model_type: str, scale: str,
                              axis_coh=None, vd_ei=None):
    """Regenerate plots/metrics/{axis_coherences,vd-ei}.png + json/vd_ei_<scale>.json
    for one saved scale directory. If ``axis_coh`` / ``vd_ei`` are not passed,
    they are loaded (axis coherence) / recomputed from npz (VD-EI)."""
    npz_path = os.path.join(scale_output_dir, 'npz', f'vectors_{scale}.npz')
    metrics_dir = os.path.join(scale_output_dir, 'plots', 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    if axis_coh is None:
        axis_coh = load_axis_coherence(scale_output_dir, scale)
    if vd_ei is None:
        vd_ei = compute_vd_ei_per_layer_from_npz(npz_path)
    if vd_ei:
        save_vd_ei(scale_output_dir, scale, vd_ei)
    if axis_coh:
        plot_axis_coherence_trajectory(
            axis_coh, scale, model_type,
            os.path.join(metrics_dir, 'axis_coherences.png'))
    if vd_ei:
        plot_vd_ei_trajectory(
            vd_ei, scale, model_type,
            os.path.join(metrics_dir, 'vd-ei.png'))


def recommend_layer(axis_coh: Dict[Tuple[str, int], dict],
                    vd_ei:    Dict[int, float],
                    total_layers: Optional[int] = None,
                    coh_tau: float = 0.85,
                    vd_window: int = 3,
                    exclude_last_frac: float = 0.20) -> dict:
    """Programmatic L* recommendation following Appendix D.3.

    Three criteria, applied as set intersections over the layer index:

    1. **Joint coherence plateau.** For each axis group (horizontal, vertical,
       distance) compute the per-axis peak coherence, then keep layers whose
       coherence is at least ``coh_tau`` of that peak. The joint plateau is
       the intersection across all three groups.
    2. **VD-EI stability.** For each layer, take the std of VD-EI over a
       window of ``±vd_window`` neighbouring layers. Keep layers whose local
       std is at or below the median local std across all layers — these are
       the "non-transient" regions.
    3. **Avoid output-specialised final layers.** Drop the last
       ``exclude_last_frac`` of the model's layers when ``total_layers`` is
       known.

    Recommendation = layer in (1) ∩ (2) ∩ (3) that maximises the *minimum*
    normalised axis coherence (i.e. the layer where every axis is closest to
    its peak). Ties broken by preferring deeper layers, consistent with the
    paper's tendency to pick the upper end of the plateau. When no layer
    survives the strictest intersection, the picker relaxes criterion (2),
    then (1). This is a heuristic starting suggestion — the paper's per-model
    picks also incorporate visual PCA inspection that this function cannot
    replicate. See `docs/LOCAL_PROBING_VERIFICATION_KO.md` for an empirical
    comparison against the four registered L* values.

    Returns a dict with the chosen layer plus per-criterion candidate sets so
    callers can inspect why a layer made (or missed) the cut.
    """
    coh_by_group: Dict[str, Dict[int, float]] = {g: {} for g in GROUP_ORDER}
    for (g, L), v in axis_coh.items():
        coh_by_group[g][L] = v['mean']

    layers_with_coh = set()
    for g in GROUP_ORDER:
        layers_with_coh.update(coh_by_group[g].keys())
    all_layers = sorted(layers_with_coh & set(vd_ei.keys()))
    if not all_layers:
        return {'recommended': None, 'reason': 'no overlapping layers'}

    # (1) per-axis plateau and joint intersection
    per_group_plateau: Dict[str, set] = {}
    for g in GROUP_ORDER:
        d = coh_by_group[g]
        if not d:
            per_group_plateau[g] = set(all_layers)
            continue
        peak = max(d.values())
        if peak <= 0:
            per_group_plateau[g] = set(all_layers)
        else:
            per_group_plateau[g] = {L for L, v in d.items() if v >= peak * coh_tau}
    joint_plateau = set(all_layers)
    for g in GROUP_ORDER:
        joint_plateau &= per_group_plateau[g]

    # (2) VD-EI local stability
    vd_layers = sorted(vd_ei.keys())
    vd_arr = np.array([vd_ei[L] for L in vd_layers])
    local_std = []
    for i in range(len(vd_layers)):
        lo, hi = max(0, i - vd_window), min(len(vd_arr), i + vd_window + 1)
        local_std.append(float(np.std(vd_arr[lo:hi])))
    stab_thresh = float(np.median(local_std))
    vd_stable = {L for L, s in zip(vd_layers, local_std) if s <= stab_thresh}

    # (3) cutoff late layers
    cutoff = None
    if total_layers is not None and total_layers > 0:
        cutoff = int(round(total_layers * (1 - exclude_last_frac)))
        not_late = {L for L in all_layers if L <= cutoff}
    else:
        not_late = set(all_layers)

    candidates = joint_plateau & vd_stable & not_late
    fallback = False
    if not candidates:
        candidates = joint_plateau & not_late
        fallback = bool(candidates)
        if not candidates:
            candidates = set(not_late) or set(all_layers)
            fallback = True

    def score(L: int) -> Tuple[float, int]:
        # primary: how close are *all three* axes to their per-axis peak;
        # tiebreak: prefer the deeper layer (matches paper's plateau-upper-end pick).
        min_norm = 1.0
        for g in GROUP_ORDER:
            if not coh_by_group[g]:
                continue
            peak = max(coh_by_group[g].values())
            if peak <= 0:
                continue
            min_norm = min(min_norm, coh_by_group[g].get(L, 0.0) / peak)
        return (min_norm, L)

    recommended = max(candidates, key=score) if candidates else None
    return {
        'recommended': recommended,
        'joint_plateau': sorted(joint_plateau),
        'vd_stable': sorted(vd_stable),
        'cutoff_layer': cutoff,
        'per_group_plateau': {g: sorted(per_group_plateau[g]) for g in GROUP_ORDER},
        'fallback_used': fallback,
    }


def plot_pca_embeddings(vectors_npz_path, scale, model_type, save_dir):
    data = np.load(vectors_npz_path, allow_pickle=True)
    layers = sorted(int(k.replace('orig_L', '')) for k in data.files if k.startswith('orig_L'))
    os.makedirs(save_dir, exist_ok=True)
    for layer in layers:
        orig   = data.get(f'orig_L{layer}')
        swap   = data.get(f'swap_L{layer}')
        labels = data.get(f'labels_L{layer}')
        deltas = data.get(f'delta_L{layer}')
        cats   = data.get(f'categories_L{layer}')
        groups = data.get(f'groups_L{layer}')
        if orig is None or len(orig) == 0:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        pca = PCA(n_components=2)
        all_pca = pca.fit_transform(np.vstack([orig, swap]))
        orig_pca = all_pca[:len(orig)]; swap_pca = all_pca[len(orig):]
        ax = axes[0]
        for cat in CATEGORY_ORDER:
            mask = np.array([str(l) == cat for l in labels])
            if mask.any():
                ax.scatter(orig_pca[mask, 0], orig_pca[mask, 1],
                           c=CAT_COLORS.get(cat, 'gray'), label=f'{cat} (orig)', alpha=0.5, s=15)
                ax.scatter(swap_pca[mask, 0], swap_pca[mask, 1],
                           c=CAT_COLORS.get(cat, 'gray'), alpha=0.5, s=15, marker='x')
        ax.set_title('Embeddings by Category\n(o=orig, x=swap)', fontsize=11)
        ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.2)
        if deltas is not None and cats is not None:
            pca_d = PCA(n_components=2)
            delta_pca = pca_d.fit_transform(deltas)
            for axi, coloring, title in [
                (axes[1], groups, 'Delta Vectors by Group'),
                (axes[2], cats,   'Delta Vectors by Category'),
            ]:
                if coloring is None:
                    continue
                color_map = GROUP_COLORS if title.endswith('Group') else CAT_COLORS
                order = GROUP_ORDER if title.endswith('Group') else CATEGORY_ORDER
                for key in order:
                    mask = np.array([str(c) == key for c in coloring])
                    if mask.any():
                        axi.scatter(delta_pca[mask, 0], delta_pca[mask, 1],
                                    c=color_map.get(key, 'gray'), label=key, alpha=0.5, s=15)
                axi.set_title(title, fontsize=11)
                axi.legend(fontsize=7 if title.endswith('Category') else 9)
                axi.grid(True, alpha=0.2)
        fig.suptitle(f'{model_type} ({scale}) – Layer {layer} – PCA', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'pca_{scale}_L{layer}.png'), dpi=200, bbox_inches='tight')
        plt.close()
    logger.info(f"Saved PCA plots to {save_dir}")


def plot_pca_3d(vectors_npz_path, scale, model_type, save_dir):
    data = np.load(vectors_npz_path, allow_pickle=True)
    layers = sorted(int(k.replace('orig_L', '')) for k in data.files if k.startswith('orig_L'))
    if not layers:
        return
    os.makedirs(save_dir, exist_ok=True)
    for layer in layers:
        deltas = data.get(f'delta_L{layer}')
        cats   = data.get(f'categories_L{layer}')
        if deltas is None or len(deltas) < 3:
            continue
        pca_d = PCA(n_components=3)
        proj  = pca_d.fit_transform(deltas)
        ev    = pca_d.explained_variance_ratio_

        fig = plt.figure(figsize=(13, 10))
        ax  = fig.add_subplot(111, projection='3d')
        if cats is not None:
            for cat in CATEGORY_ORDER:
                mask = np.array([str(c) == cat for c in cats])
                if mask.any():
                    ax.scatter(proj[mask, 0], proj[mask, 1], proj[mask, 2],
                               c=CAT_COLORS.get(cat, 'gray'), label=cat, alpha=0.55, s=30)
        ax.set_title('Delta Vectors by Category', fontsize=22, pad=12)
        ax.set_xlabel(f'PC1 ({ev[0]:.1%})', fontsize=18, labelpad=25)
        ax.set_ylabel(f'PC2 ({ev[1]:.1%})', fontsize=18, labelpad=25)
        ax.set_zlabel(''); ax.tick_params(axis='both', labelsize=14)
        ax.legend(fontsize=16, ncol=2, loc='upper right')
        fig.canvas.draw()
        ax_pos = ax.get_position()
        fig.text(ax_pos.x1 + 0.04, (ax_pos.y0 + ax_pos.y1) / 2,
                 f'PC3 ({ev[2]:.1%})', fontsize=18, va='center', ha='center', rotation=90)
        fig.suptitle(f'{model_type} ({scale}) — L{layer}', fontsize=24, fontweight='bold',
                     x=(ax_pos.x0 + ax_pos.x1) / 2, y=1.01)
        plt.savefig(os.path.join(save_dir, f'pca_{scale}_L{layer}.png'),
                    dpi=200, bbox_inches='tight', pad_inches=0.5)
        plt.close()
    logger.info(f"Saved 3D PCA plots to {save_dir}")


def run_all_layer_heatmaps(model_dir: str, model_type: str, scales: List[str]):
    for scale in scales:
        npz_path = os.path.join(model_dir, 'npz', f'vectors_{scale}.npz')
        if not os.path.exists(npz_path):
            logger.warning(f'[{model_type}/{scale}] NPZ not found, skipping heatmaps.')
            continue
        data = np.load(npz_path, allow_pickle=True)
        npz_layers = sorted(int(k.replace('orig_L', '')) for k in data.files if k.startswith('orig_L'))
        data.close()
        csv_dir = os.path.join(model_dir, 'csv')
        out_dir = os.path.join(model_dir, 'plots', 'heatmap')
        os.makedirs(out_dir, exist_ok=True)
        for layer in npz_layers:
            csv_path = os.path.join(csv_dir, f'delta_similarity_{scale}_L{layer}.csv')
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path, index_col=0)
            available = [c for c in CATEGORY_ORDER if c in df.index]
            if not available:
                continue
            plot_delta_heatmap(
                df.loc[available, available],
                f'{model_type} ({scale}) — Delta Similarity Heatmap L{layer}',
                os.path.join(out_dir, f'heatmap_{scale}_L{layer}.png'),
            )


def run_all_layer_pca(model_dir: str, model_type: str, scales: List[str]):
    for scale in scales:
        npz_path = os.path.join(model_dir, 'npz', f'vectors_{scale}.npz')
        if not os.path.exists(npz_path):
            continue
        for plot_fn, sub2 in [(plot_pca_embeddings, 'pca'), (plot_pca_3d, 'pca_3d')]:
            out = os.path.join(model_dir, 'plots', sub2)
            os.makedirs(out, exist_ok=True)
            plot_fn(npz_path, scale, model_type, out)


# ============================================================================
# Main Pipeline
# ============================================================================

def process_scale(args, model_type: str, scale: str, swap_pairs: List[dict]):
    """Run full inference + analysis for one (model_type, scale) checkpoint."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {model_type} / {scale}")
    logger.info(f"{'='*60}")

    extractor     = get_extractor(model_type, scale, device=args.device)
    target_layers = extractor.target_layers

    output_dir = os.path.join(args.output_dir, _model_key(model_type, scale))
    plots_dir  = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # ── Extraction ────────────────────────────────────────────────────────────
    logger.info("\n--- Extraction ---")
    swap_records = extract_swap_features(
        extractor, swap_pairs, max_samples_per_category=args.max_samples_per_category)

    # ── Analysis ──────────────────────────────────────────────────────────────
    logger.info("\n--- Analysis ---")
    axis_coh = compute_axis_coherence(swap_records, target_layers)
    vd_ei    = compute_vd_ei_per_layer(swap_records, target_layers)
    delta_heatmaps = {layer: compute_delta_similarity_matrix(swap_records, layer)
                      for layer in target_layers}

    max_layer = max(target_layers)
    for group in GROUP_ORDER:
        if (group, max_layer) in axis_coh:
            v = axis_coh[(group, max_layer)]
            logger.info(f"  axis coherence [{group}, L{max_layer}]: {v['mean']:.4f} ± {v['std']:.4f}")
    if max_layer in vd_ei:
        logger.info(f"  VD-EI [L{max_layer}]: {vd_ei[max_layer]:+.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    logger.info("\n--- Saving ---")
    save_vectors_npz(scale, swap_records, target_layers, output_dir)
    save_scale_results(scale, swap_records, axis_coh, delta_heatmaps, output_dir)
    save_vd_ei(output_dir, scale, vd_ei)

    # ── Plots ─────────────────────────────────────────────────────────────────
    logger.info("\n--- Plots ---")
    rebuild_metrics_for_scale(output_dir, model_type, scale,
                              axis_coh=axis_coh, vd_ei=vd_ei)
    npz_path = os.path.join(output_dir, 'npz', f'vectors_{scale}.npz')
    if os.path.exists(npz_path):
        run_all_layer_pca(output_dir, model_type, [scale])
    run_all_layer_heatmaps(output_dir, model_type, [scale])

    del swap_records
    extractor.cleanup()
    logger.info(f"\n  Scale {scale} complete.")


def _scale_dir(output_dir: str, model_type: str, scale: str) -> str:
    return os.path.join(output_dir, _model_key(model_type, scale))


def run_merge(args, model_type: str):
    """Load saved per-scale data and generate cross-scale comparison plots."""
    spec = MODEL_REGISTRY.get(model_type)
    is_merge_only = model_type in MERGE_CONFIGS

    if is_merge_only:
        mc          = MERGE_CONFIGS[model_type]
        scale_order = mc['scale_order']
        scale_src   = mc['scale_sources']
    else:
        scale_order = list(spec.checkpoints.keys()) if spec else []
        scale_src   = {s: model_type for s in scale_order}

    available_scales = [s for s in scale_order if s in args.scales]
    logger.info(f"Merging scales: {available_scales}")

    group_name = args.group_name or model_type
    merge_out  = args.merge_output_dir or os.path.join(
        os.path.dirname(args.output_dir.rstrip('/')), 'compare', group_name)
    plots_dir  = os.path.join(merge_out, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    def _sd(scale):
        src = scale_src.get(scale, model_type)
        return _scale_dir(args.output_dir, src, scale)

    all_coh = {}; all_dh = {}; all_vdei = {}

    for scale in available_scales:
        sd  = _sd(scale)
        coh = load_axis_coherence(sd, scale)
        dh  = load_delta_heatmaps(sd, scale)
        vd  = load_vd_ei(sd, scale)
        if not vd:
            # Fallback: recompute from npz so older saved_data still merges.
            vd = compute_vd_ei_per_layer_from_npz(
                os.path.join(sd, 'npz', f'vectors_{scale}.npz'))
            if vd:
                save_vd_ei(sd, scale, vd)
        if coh: all_coh[scale]   = coh
        if dh:  all_dh[scale]    = dh
        if vd:  all_vdei[scale]  = vd
        logger.info(f"  Loaded data for '{scale}'")

    metrics_dir = os.path.join(plots_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    if len(all_coh) > 1:
        plot_cross_scale_axis_coherence(
            all_coh, model_type,
            os.path.join(metrics_dir, 'axis_coherences.png'))
    if len(all_vdei) > 1:
        plot_cross_scale_vd_ei(
            all_vdei, model_type,
            os.path.join(metrics_dir, 'vd-ei.png'))

    logger.info(f"\n=== Merge complete. Results in: {merge_out} ===")


# ============================================================================
# CLI
# ============================================================================

def main():
    all_model_types = list(MODEL_REGISTRY.keys()) + list(MERGE_CONFIGS.keys())

    parser = argparse.ArgumentParser(
        description='probing.py — Swap Analysis for Spatial Representations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Registered models:
  """ + ', '.join(MODEL_REGISTRY.keys()) + """

Merge configs:
  """ + (', '.join(MERGE_CONFIGS.keys()) if MERGE_CONFIGS else '(none)') + """

Examples:
  # Run a single model
  python probing.py --model_type qwen25 --scales vanilla

  # Run all checkpoints of a model family
  python probing.py --model_type qwen25

  # Generate cross-scale comparison plots from saved results
  python probing.py --model_type qwen25 --merge
""")

    parser.add_argument('--data_path', type=str,
                        default='./data/EmbSpatial-Bench.tsv',
                        help="Path to the EmbSpatial-Bench TSV file. "
                             "Download with: "
                             "`huggingface-cli download ch-min/EmbSpatial-Bench-tsv "
                             "EmbSpatial-Bench.tsv --repo-type dataset --local-dir ./data`.")
    parser.add_argument('--model_type', type=str, required=True,
                        choices=all_model_types,
                        help='Model family to run (see registered models above).')
    parser.add_argument('--scales', type=str, nargs='+', default=None,
                        help='Scale names to process (default: all for the given model_type).')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Root directory for saved results. '
                             'Defaults to <script_dir>/results/saved_data.')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--merge', action='store_true',
                        help='Merge mode: generate cross-scale plots from saved data.')
    parser.add_argument('--rebuild-metrics', action='store_true', dest='rebuild_metrics',
                        help='Skip inference; recompute VD-EI from saved npz and regenerate '
                             'plots/metrics/{axis_coherences,vd-ei}.png plus '
                             'json/vd_ei_<scale>.json for every requested scale. '
                             'Iterates over scales from MODEL_REGISTRY (or --scales) only; '
                             'on-disk dirs not in the registry are not auto-discovered — '
                             'pass them via --scales explicitly.')
    parser.add_argument('--recommend-layer', action='store_true', dest='recommend_layer_flag',
                        help='Skip inference; print the systematic L* recommendation from '
                             '`recommend_layer` (App. D.3 criteria — joint coherence plateau, '
                             'VD-EI stability, exclude last 20%% of layers) for every saved '
                             'scale and compare against MODEL_REGISTRY[...].paper_layer.')
    parser.add_argument('--merge-output-dir', type=str, default=None, dest='merge_output_dir',
                        help='Override the output directory for cross-scale plots.')
    parser.add_argument('--group-name', type=str, default=None, dest='group_name',
                        help='Folder name under compare/ for merged output.')
    parser.add_argument('--max-samples-per-category', type=int, default=200,
                        dest='max_samples_per_category',
                        help='Maximum swap pairs per spatial category (0 = no limit).')

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(_HERE, 'results', 'saved_data')
    log_dir = os.path.join(_HERE, 'results', 'logs')

    # Validate merge-only types
    if args.model_type in MERGE_CONFIGS and not args.merge:
        parser.error(
            f"'{args.model_type}' is a merge-only config. Add --merge to run it.\n"
            f"  Example: python probing.py --model_type {args.model_type} --merge"
        )

    # Default scales
    if args.scales is None:
        if args.model_type in MERGE_CONFIGS:
            args.scales = MERGE_CONFIGS[args.model_type]['scale_order']
        elif args.model_type in MODEL_REGISTRY:
            args.scales = list(MODEL_REGISTRY[args.model_type].checkpoints.keys())
        else:
            args.scales = []

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # ── Merge mode ────────────────────────────────────────────────────────────
    if args.merge:
        group_name = args.group_name or args.model_type
        log_path   = _setup_file_logging(group_name, log_dir)
        logger.info(f"Logging to: {log_path}")
        logger.info("\n=== MERGE MODE ===")
        run_merge(args, args.model_type)
        return

    # ── Rebuild-metrics mode ──────────────────────────────────────────────────
    if args.rebuild_metrics:
        log_path = _setup_file_logging(args.model_type + "_rebuild", log_dir)
        logger.info(f"Logging to: {log_path}")
        logger.info("\n=== REBUILD-METRICS MODE ===")
        spec = MODEL_REGISTRY.get(args.model_type)
        scales = args.scales or (list(spec.checkpoints.keys()) if spec else [])
        for scale in scales:
            sd = os.path.join(args.output_dir, _model_key(args.model_type, scale))
            if not os.path.isdir(sd):
                logger.warning(f"  {sd} not found, skipping.")
                continue
            logger.info(f"  Rebuilding metrics for {args.model_type}/{scale}")
            rebuild_metrics_for_scale(sd, args.model_type, scale)
        return

    # ── Recommend-layer mode ──────────────────────────────────────────────────
    if args.recommend_layer_flag:
        log_path = _setup_file_logging(args.model_type + "_recommend", log_dir)
        logger.info(f"Logging to: {log_path}")
        logger.info("\n=== RECOMMEND-LAYER MODE ===")
        spec = MODEL_REGISTRY.get(args.model_type)
        scales = args.scales or (list(spec.checkpoints.keys()) if spec else [])
        paper_L = spec.paper_layer if spec else None
        total = spec.total_layers if spec else None
        logger.info(f"  Paper-registered L* for {args.model_type}: "
                    f"{('L'+str(paper_L)) if paper_L is not None else '(not set)'}  "
                    f"(total_layers={total})")
        for scale in scales:
            sd = os.path.join(args.output_dir, _model_key(args.model_type, scale))
            if not os.path.isdir(sd):
                logger.warning(f"  {sd} not found, skipping.")
                continue
            coh = load_axis_coherence(sd, scale)
            vd  = load_vd_ei(sd, scale)
            if not vd:
                # backfill from npz if user hasn't rebuilt yet
                vd = compute_vd_ei_per_layer_from_npz(
                    os.path.join(sd, 'npz', f'vectors_{scale}.npz'))
            if not coh or not vd:
                logger.warning(f"  {scale}: missing axis_coh or vd_ei, skipping.")
                continue
            rec = recommend_layer(coh, vd, total_layers=total)
            L = rec['recommended']
            jp = rec['joint_plateau']
            jp_str = f"L{jp[0]}-L{jp[-1]}" if jp else "(empty → fallback)"
            delta = f"  Δ{L - paper_L:+d}" if (L is not None and paper_L is not None) else ""
            logger.info(f"  {scale:<10s}  recommended L*={'L'+str(L) if L is not None else '?'}"
                        f"{delta}   joint plateau={jp_str}   "
                        f"fallback={rec['fallback_used']}")
        return

    # ── Inference mode ────────────────────────────────────────────────────────
    logger.info("\n=== Loading swap pairs ===")
    swap_pairs = load_swap_pairs(args.data_path, args.seed)

    spec = MODEL_REGISTRY[args.model_type]
    for scale in args.scales:
        if scale not in spec.checkpoints:
            logger.warning(f"Scale '{scale}' not in MODEL_REGISTRY['{args.model_type}'].checkpoints, skipping.")
            continue

        vlm_key  = _model_key(args.model_type, scale)
        log_path = _setup_file_logging(vlm_key, log_dir)
        logger.info(f"Logging to: {log_path}")

        raw_path = spec.checkpoints[scale]
        if os.path.isabs(raw_path) and not os.path.exists(raw_path):
            logger.warning(f"Model path not found: {raw_path} (scale='{scale}'), skipping.")
            continue

        try:
            process_scale(args, args.model_type, scale, swap_pairs)
        except Exception as e:
            import traceback
            logger.error(f"Failed {args.model_type}/{scale}: {e}")
            traceback.print_exc()

    logger.info(f"\n{'='*60}")
    logger.info("=== All scales complete ===")
    logger.info(f"Results: {args.output_dir}")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
