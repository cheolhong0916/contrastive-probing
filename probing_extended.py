#!/usr/bin/env python3
"""
probing.py — Swap Analysis: Minimal Pair Probing for Spatial Representations
=============================================================================

Creates minimal pairs by swapping obj1 <-> obj2 in spatial questions:
  Original : "Is A to the left or right of B?"  → left
  Swapped  : "Is B to the left or right of A?"  → right

Spatial categories
------------------
  horizontal : left / right
  vertical   : above / below
  distance   : far / close

Analyses produced
-----------------
  1. Delta vectors            : delta = hidden(swapped) - hidden(original)
  2. Within-category consistency  : do all left→right swaps point the same way?
  3. Sign-corrected consistency   : align opposite categories before comparing
  4. Cross-group alignment        : cos(Δ_vertical, Δ_distance) — perspective bias
  5. Delta similarity heatmap     : 6×6 cosine similarity of mean Δ per category
  6. Prediction accuracy charts   : orig / swap / both-correct per group & scale
  7. PCA visualisations           : 2-D and 3-D PCA of embeddings / delta vectors

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO ADD A NEW MODEL  (two steps)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1 — Implement an extractor class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Inherit from BaseHiddenStateExtractor and implement four methods:

    class MyModelExtractor(BaseHiddenStateExtractor):
        def _load_model(self):
            # Load self.model and self.processor (or equivalent)
            ...

        def _get_num_layers(self) -> int:
            # Return the number of transformer layers
            return len(self.model.language_model.model.layers)

        def _get_layer_module(self, layer_idx: int):
            # Return the nn.Module for layer layer_idx
            return self.model.language_model.model.layers[layer_idx]

        def extract_and_predict(self, image: Image.Image, question: str):
            # Run inference; hidden states are captured automatically via hooks.
            # Return (self.hidden_states.copy(), answer_string)
            self.hidden_states = {}
            ...
            return self.hidden_states.copy(), answer

The framework registers a forward hook on every target layer automatically;
you do *not* need to handle hook registration yourself.

STEP 2 — Register the model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Add an entry to MODEL_REGISTRY (search for "MODEL_REGISTRY = {" below):

    MODEL_REGISTRY["my_model"] = ModelSpec(
        extractor_class = MyModelExtractor,
        checkpoints = {
            "base"       : "org/model-name-on-huggingface",
            "finetuned"  : "/local/path/to/checkpoint",
        },
        display_name = "My Model",          # used in plot titles
        plot_color   = "#17becf",           # optional hex colour for plots
    )

Then run:
    python probing.py --model_type my_model --scales base

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # Run a single registered model
  python probing.py --model_type qwen25 --scales 3b

  # Run all checkpoints for a model family
  python probing.py --model_type prismatic

  # Generate cross-scale comparison plots (after inference)
  python probing.py --model_type qwen25 --merge

  # Combine two families into one comparison plot
  python probing.py --model_type qwen_all --merge
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

SHORT_OPPOSITE_MAP = {
    'left': 'right', 'right': 'left',
    'above': 'below', 'below': 'above',
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

SYNONYMS = {
    'below': ['under', 'beneath'],
    'close': ['near', 'nearby'],
    'far':   ['distant'],
}

# ── Question templates ────────────────────────────────────────────────────────

_Q_TAIL_MCQ = "Answer with a single letter A or B."

MCQ_TEMPLATES = {
    'horizontal': {
        'left_first':  "Is the {obj1} to the left or right of the {obj2}? (A) left  (B) right  " + _Q_TAIL_MCQ,
        'right_first': "Is the {obj1} to the left or right of the {obj2}? (A) right  (B) left  " + _Q_TAIL_MCQ,
    },
    'vertical': {
        'above_first': "Is the {obj1} above or below the {obj2}? (A) above  (B) below  " + _Q_TAIL_MCQ,
        'below_first': "Is the {obj1} above or below the {obj2}? (A) below  (B) above  " + _Q_TAIL_MCQ,
    },
    'distance': {
        'far_first':   "Compared to {ref}, is {subj} far or close from you? (A) far  (B) close  " + _Q_TAIL_MCQ,
        'close_first': "Compared to {ref}, is {subj} far or close from you? (A) close  (B) far  " + _Q_TAIL_MCQ,
    },
}

MCQ_LETTER = {
    'horizontal': {
        'left_first':  {'left': 'a', 'right': 'b'},
        'right_first': {'left': 'b', 'right': 'a'},
    },
    'vertical': {
        'above_first': {'above': 'a', 'below': 'b'},
        'below_first': {'above': 'b', 'below': 'a'},
    },
    'distance': {
        'far_first':   {'far': 'a', 'close': 'b'},
        'close_first': {'far': 'b', 'close': 'a'},
    },
}

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
    """
    extractor_class: Type[Any]
    checkpoints: Dict[str, str]
    display_name: str = ''
    base_processor_id: str = ''
    plot_color: str = _DEFAULT_SCALE_COLOR


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
):
    """Class decorator that registers a VLM extractor in MODEL_REGISTRY.

    Example
    -------
    @register_model(
        model_type   = "my_model",
        checkpoints  = {"base": "org/my-model-7b"},
        display_name = "My Model 7B",
        plot_color   = "#e377c2",
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

# ── Helper shared by Molmo2 / Qwen3 / Prismatic ───────────────────────────────

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
        logger.info(f"Loaded Qwen2.5-VL from {self.model_path}")

    def _get_num_layers(self) -> int:
        return len(self.model.model.layers)

    def _get_layer_module(self, layer_idx: int):
        return self.model.model.layers[layer_idx]

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


# ── PrismaticVLM (TRI-ML) ─────────────────────────────────────────────────────

class PrismaticExtractor(BaseHiddenStateExtractor):
    """Extractor for TRI-ML PrismaticVLM family.

    Supports models hosted on HuggingFace under the TRI-ML organisation:
      - TRI-ML/prism-dinosiglip-224px+7b
      - TRI-ML/prism-dinosiglip-336px+7b
      - TRI-ML/prism-clip+7b
      (and any fine-tuned checkpoints following the same architecture)

    The extractor uses the standard HuggingFace AutoModelForVision2Seq interface
    with trust_remote_code=True, which is how PrismaticVLM is packaged on the Hub.
    Hidden states are captured from the LLM backbone's transformer layers.
    """

    def _load_model(self):
        from transformers import AutoProcessor, AutoModelForVision2Seq
        proc_id = self.base_processor_id or self.model_path
        self.processor = AutoProcessor.from_pretrained(proc_id, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map='auto',
        ).eval()
        self.llm_layers = _find_llm_layers(self.model, [
            # Common Prismatic path (Llama-based LLM backbone)
            ['llm_backbone', 'model', 'layers'],
            ['language_model', 'model', 'layers'],
            ['model', 'language_model', 'model', 'layers'],
            ['model', 'layers'],
        ], hint='PrismaticVLM')
        if self.llm_layers is None:
            raise ValueError(
                "Could not find transformer layers in PrismaticVLM model. "
                "Please check the model architecture and update PrismaticExtractor."
            )
        logger.info(f"Loaded PrismaticVLM from {self.model_path}")

    def _get_num_layers(self) -> int:
        return len(self.llm_layers)

    def _get_layer_module(self, layer_idx: int):
        return self.llm_layers[layer_idx]

    def extract_and_predict(self, image, question):
        self.hidden_states = {}
        # Build a single-turn conversation.
        # PrismaticVLM processors expose apply_chat_template when loaded via HF.
        if hasattr(self.processor, 'apply_chat_template'):
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ]}]
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(images=[image], text=prompt, return_tensors="pt")
        else:
            # Fallback: plain text prompt (some older checkpoints)
            inputs = self.processor(images=[image], text=question, return_tensors="pt")

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=20, do_sample=False)
        input_len = inputs['input_ids'].shape[1]
        answer = self.processor.tokenizer.decode(
            output_ids[0, input_len:], skip_special_tokens=True
        ).strip()
        return self.hidden_states.copy(), answer


# ── PrismaticVLM — native loader (prismatic package) ─────────────────────────

class PrismaticNativeExtractor(BaseHiddenStateExtractor):
    """Extractor for TRI-ML PrismaticVLM using the native *prismatic* package.

    Preferred over PrismaticExtractor when running in the *prismatic-vlms*
    conda environment because it avoids the huggingface_hub validation error
    caused by the '+' character in model repo IDs (e.g.
    'prism-dinosiglip-224px+7b').

    The model_path / checkpoint key should be the prismatic model ID string,
    not a HuggingFace URL, e.g.: "prism-dinosiglip-224px+7b".

    Requires:  conda env prismatic-vlms  (or pip install prismatic-vlms)

    Hidden states are captured from the Llama-2 language backbone at
      model.llm_backbone.llm.model.layers
    """

    def _load_model(self):
        try:
            import prismatic as _prismatic
        except ImportError:
            raise ImportError(
                "prismatic package not found. "
                "Run inside the 'prismatic-vlms' conda environment, or:\n"
                "  pip install prismatic-vlms"
            )
        model_id = self.model_path   # e.g. "prism-dinosiglip-224px+7b"
        logger.info(f"Loading PrismaticVLM '{model_id}' via native prismatic package…")
        self.vlm = _prismatic.load(model_id)
        self.vlm.to(self.device).eval()

        # Expose the language model layers for hook registration.
        # LLaMa2LLMBackbone stores the HF LlamaForCausalLM at .llm
        llm_backbone = self.vlm.llm_backbone
        self.llm_layers = _find_llm_layers(llm_backbone, [
            ['llm', 'model', 'layers'],
            ['model', 'layers'],
        ], hint='PrismaticNative')
        if self.llm_layers is None:
            raise ValueError(
                "Could not find transformer layers in native PrismaticVLM. "
                "Check the llm_backbone attribute path."
            )
        logger.info(f"PrismaticVLM '{model_id}': {len(self.llm_layers)} layers.")

        # Expose .model for the cleanup helper (expects model attribute)
        self.model = self.vlm

    def _get_num_layers(self) -> int:
        return len(self.llm_layers)

    def _get_layer_module(self, layer_idx: int):
        return self.llm_layers[layer_idx]

    def extract_and_predict(self, image, question):
        self.hidden_states = {}
        with torch.no_grad():
            answer = self.vlm.generate(image, question)
        return self.hidden_states.copy(), answer.strip()


# ── OpenVLA (openvla/openvla-7b) ──────────────────────────────────────────────

class OpenVLAExtractor(BaseHiddenStateExtractor):
    """Extractor for OpenVLA (openvla/openvla-7b).

    OpenVLA is fine-tuned from PrismaticVLM (DINOv2 + SigLIP + Llama 2 7B)
    on 970k Open X-Embodiment robot demonstration episodes.

    The model predicts continuous robot actions (not text), so the answer
    returned by extract_and_predict is always "N/A". Accuracy metrics will
    be zero — the value of this extractor lies in the **hidden-state
    comparison** against the base PrismaticVLM checkpoint.

    Loading: AutoModelForCausalLM + trust_remote_code=True (same interface
    as PrismaticVLM). Requires the same environment as PrismaticExtractor.
    """

    def _load_model(self):
        # OpenVLA's config.json auto_map uses AutoModelForVision2Seq,
        # not AutoModelForCausalLM.
        from transformers import AutoModelForVision2Seq, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True,
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            low_cpu_mem_usage=True,
        ).eval()
        self.llm_layers = _find_llm_layers(self.model, [
            ['language_model', 'model', 'layers'],
            ['llm_backbone', 'llm', 'model', 'layers'],
            ['model', 'language_model', 'model', 'layers'],
            ['model', 'layers'],
        ], hint='OpenVLA')
        if self.llm_layers is None:
            raise ValueError("Could not find transformer layers in OpenVLA model.")
        logger.info(f"Loaded OpenVLA from {self.model_path}")

    def _get_num_layers(self) -> int:
        return len(self.llm_layers)

    def _get_layer_module(self, layer_idx: int):
        return self.llm_layers[layer_idx]

    def extract_and_predict(self, image, question):
        """Run one forward pass to capture LLM hidden states.

        OpenVLA generates robot action tokens, not spatial words.
        We trigger exactly one forward (prefill) pass with max_new_tokens=1
        so that the hooks capture the last-token hidden state of the input.
        The returned answer is always 'N/A'.
        """
        self.hidden_states = {}
        # OpenVLA processor: processor(text, image)
        inputs = self.processor(question, image, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            try:
                self.model.generate(**inputs, max_new_tokens=1, do_sample=False)
            except Exception as e:
                logger.warning(f"OpenVLA generate() failed ({e}); falling back to forward().")
                self.model(**inputs)
        # OpenVLA outputs action tokens — accuracy metrics will be N/A
        return self.hidden_states.copy(), "N/A"


# ── PaliGemma (google/paligemma-3b-pt-224) ───────────────────────────────────

class PaliGemmaExtractor(BaseHiddenStateExtractor):
    """Extractor for Google PaliGemma (SigLIP + Gemma 2B, 3B total).

    Works with any PaliGemma checkpoint:
      - google/paligemma-3b-pt-224   (pretrained)
      - google/paligemma-3b-mix-224  (fine-tuned on VQA mix)

    The language-model backbone is Gemma 2B; hidden states are extracted
    from its transformer layers (model.language_model.model.layers).
    """

    def _load_model(self):
        from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
        proc_id = self.base_processor_id or self.model_path
        self.processor = AutoProcessor.from_pretrained(proc_id)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto',
        ).eval()
        self.llm_layers = _find_llm_layers(self.model, [
            ['language_model', 'layers'],           # transformers 4.57: .language_model = GemmaModel
            ['language_model', 'model', 'layers'],  # older: .language_model = GemmaForCausalLM
            ['model', 'language_model', 'layers'],
            ['model', 'language_model', 'model', 'layers'],
        ], hint='PaliGemma')
        if self.llm_layers is None:
            raise ValueError("Could not find transformer layers in PaliGemma model.")
        logger.info(f"Loaded PaliGemma from {self.model_path}")

    def _get_num_layers(self) -> int:
        return len(self.llm_layers)

    def _get_layer_module(self, layer_idx: int):
        return self.llm_layers[layer_idx]

    def extract_and_predict(self, image, question):
        self.hidden_states = {}
        inputs = self.processor(images=image, text=question, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=20, do_sample=False)
        input_len = inputs['input_ids'].shape[1]
        answer = self.processor.tokenizer.decode(
            output_ids[0, input_len:], skip_special_tokens=True
        ).strip()
        return self.hidden_states.copy(), answer


# ── π0 / Pi0 (lerobot/pi0_base) ──────────────────────────────────────────────

class Pi0Extractor(BaseHiddenStateExtractor):
    """Extractor for π0 (Physical Intelligence) loaded via LeRobot.

    π0 fine-tunes PaliGemma (3B) together with a flow-matching action expert
    on a mix of Open X-Embodiment data and in-house robot trajectories.
    This extractor isolates the **PaliGemma backbone** from the π0 checkpoint
    and probes its spatial representations via standard VQA inference.

    Requires:  pip install lerobot

    Checkpoints
    -----------
    lerobot/pi0_base  : base π0 policy
    lerobot/pi05_base : π0.5 (extended training)

    Notes
    -----
    - If the PaliGemma backbone was frozen during VLA training the hidden
      states will match the base PaliGemma.  Use a full-finetune checkpoint
      (e.g. pi0_base without train_expert_only) for a meaningful comparison.
    - The image processor is loaded from the base PaliGemma checkpoint
      (set via base_processor_id, default: google/paligemma-3b-pt-224).
    """

    @staticmethod
    def _patch_transformers_siglip_check():
        """Inject a stub siglip.check module so lerobot pi0 loads with
        standard (non-openpi-patched) transformers.  The patch is harmless:
        lerobot uses it only to assert the patched transformers is installed;
        the actual PaliGemma / Gemma model code works with stock transformers.
        """
        import sys
        from types import ModuleType
        mod_name = 'transformers.models.siglip.check'
        if mod_name not in sys.modules:
            stub = ModuleType(mod_name)
            stub.check_whether_transformers_replace_is_installed_correctly = lambda: True
            import transformers.models.siglip as _siglip
            _siglip.check = stub
            sys.modules[mod_name] = stub

    def _load_model(self):
        # lerobot pi0 asserts a custom-patched transformers (openpi fork).
        # Inject a stub so stock transformers passes the check.
        self._patch_transformers_siglip_check()

        try:
            # lerobot >= 0.4: lerobot.policies.*
            # lerobot <  0.4: lerobot.common.policies.*
            try:
                from lerobot.policies.pi0.modeling_pi0 import PI0Policy
            except ImportError:
                from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
        except ImportError:
            raise ImportError(
                "lerobot package not found. Install with:\n"
                "  pip install lerobot\n"
                "See: https://github.com/huggingface/lerobot"
            )

        logger.info(f"Loading π0 policy from {self.model_path} via lerobot …")
        policy = PI0Policy.from_pretrained(self.model_path)
        policy.eval()

        paligemma, tokenizer = self._extract_paligemma(policy)
        self.model = paligemma.to(torch.bfloat16).to('cuda').eval()
        self.tokenizer = tokenizer

        # transformers may cache torch_dtype as a string at __init__ time
        # (e.g. PaliGemmaModel.text_config_dtype = config.get_text_config().dtype
        # returns 'float32'). torch.finfo() requires an actual torch.dtype.
        # Patch the live attribute on PaliGemmaModel directly.
        pg_inner = getattr(self.model, 'model', None)  # PaliGemmaModel inside PaliGemmaForConditionalGeneration
        if pg_inner is not None and hasattr(pg_inner, 'text_config_dtype'):
            # Must match actual weight dtype (bfloat16), not the config string.
            pg_inner.text_config_dtype = next(self.model.parameters()).dtype

        # Image processor from base PaliGemma
        from transformers import AutoProcessor
        proc_id = self.base_processor_id or "google/paligemma-3b-pt-224"
        self.processor = AutoProcessor.from_pretrained(proc_id)

        self.llm_layers = _find_llm_layers(self.model, [
            # transformers 4.57: PaliGemmaForConditionalGeneration.language_model
            # is a property returning PaliGemmaModel.language_model (GemmaModel),
            # which has .layers directly (not .model.layers).
            ['language_model', 'layers'],
            ['language_model', 'model', 'layers'],
            ['model', 'language_model', 'layers'],
            ['model', 'language_model', 'model', 'layers'],
        ], hint='Pi0-PaliGemma')
        if self.llm_layers is None:
            raise ValueError(
                "Could not find transformer layers in π0's PaliGemma backbone. "
                "Check the lerobot version and model structure."
            )
        logger.info(f"π0 PaliGemma backbone: {len(self.llm_layers)} transformer layers.")

    def _extract_paligemma(self, policy):
        """Walk π0 policy attributes to find the PaliGemma backbone."""
        pi0 = policy.model  # PI0 nn.Module

        # ── Try known attribute paths (lerobot versions vary) ─────────────────
        pg_paths = [
            ['paligemma_with_expert', 'paligemma'],
            ['paligemma'],
            ['vlm'],
        ]
        paligemma = None
        for path in pg_paths:
            obj = pi0
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None and hasattr(obj, 'language_model'):
                paligemma = obj
                logger.info(f"Pi0: PaliGemma found at model.{'.'.join(path)}")
                break

        if paligemma is None:
            # Fallback: scan all named modules for a PaliGemma class
            for name, module in pi0.named_modules():
                cls_name = type(module).__name__
                if cls_name in ('PaliGemmaForConditionalGeneration',
                                'PaliGemmaModel'):
                    paligemma = module
                    logger.info(f"Pi0: PaliGemma found via scan at '{name}'")
                    break

        if paligemma is None:
            raise ValueError(
                "Could not find PaliGemma backbone in π0 model. "
                "Attribute paths tried: " + str(pg_paths)
            )

        # ── Tokenizer ─────────────────────────────────────────────────────────
        tokenizer = None
        for path in [['paligemma_with_expert', 'tokenizer'], ['tokenizer']]:
            obj = pi0
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None and hasattr(obj, 'decode'):
                tokenizer = obj
                break

        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.base_processor_id or "google/paligemma-3b-pt-224"
            )

        return paligemma, tokenizer

    def _get_num_layers(self) -> int:
        return len(self.llm_layers)

    def _get_layer_module(self, layer_idx: int):
        return self.llm_layers[layer_idx]

    def extract_and_predict(self, image, question):
        self.hidden_states = {}
        inputs = self.processor(images=image, text=question, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            try:
                output_ids = self.model.generate(**inputs, max_new_tokens=20, do_sample=False)
                input_len = inputs['input_ids'].shape[1]
                answer = self.tokenizer.decode(
                    output_ids[0, input_len:], skip_special_tokens=True
                ).strip()
            except Exception as e:
                logger.warning(
                    f"Pi0 generate() failed ({e}); doing forward-only pass for hidden states."
                )
                try:
                    self.model(**inputs)
                except Exception as e2:
                    logger.warning(f"Pi0 forward() also failed ({e2}); no hidden states for this sample.")
                answer = "N/A"
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
        "3b"  : "Qwen/Qwen2.5-VL-3B-Instruct",
        # Add fine-tuned checkpoints here, e.g.:
        # "ft_80k" : "/path/to/qwen25_ft_80k",
    },
    display_name      = "Qwen2.5-VL",
    base_processor_id = "Qwen/Qwen2.5-VL-3B-Instruct",
    plot_color        = "#1f77b4",
)

MODEL_REGISTRY["qwen3"] = ModelSpec(
    extractor_class   = Qwen3VLExtractor,
    checkpoints       = {
        "32b"  : "Qwen/Qwen3-VL-32B-Instruct",
        "235b" : "Qwen/Qwen3-VL-235B-A22B-Instruct",
    },
    display_name = "Qwen3-VL",
    plot_color   = "#bcbd22",
)

MODEL_REGISTRY["molmo"] = ModelSpec(
    extractor_class = MolmoExtractor,
    checkpoints     = {
        "7b" : "allenai/Molmo-7B-O-0924",
    },
    display_name = "Molmo-7B",
    plot_color   = "#ff7f0e",
)

MODEL_REGISTRY["molmo2"] = ModelSpec(
    extractor_class = Molmo2Extractor,
    checkpoints     = {
        "8b" : "allenai/Molmo2-8B",
    },
    display_name = "Molmo2-8B",
    plot_color   = "#17becf",
)

MODEL_REGISTRY["nvila"] = ModelSpec(
    extractor_class = NVILAExtractor,
    checkpoints     = {
        "2b" : "Efficient-Large-Model/NVILA-Lite-2B",
        # Add fine-tuned checkpoints here.
    },
    display_name = "NVILA-Lite",
    plot_color   = "#2ca02c",
)

MODEL_REGISTRY["prismatic"] = ModelSpec(
    # Uses the native prismatic package (prismatic-vlms env).
    # model_path = native model ID string, NOT a HuggingFace repo URL.
    # The '+' in model IDs like 'prism-dinosiglip-224px+7b' is not handled
    # by older huggingface_hub; the native loader sidesteps this entirely.
    extractor_class = PrismaticNativeExtractor,
    checkpoints     = {
        # Base VLM — same checkpoint OpenVLA was fine-tuned from.
        "7b-dino"  : "prism-dinosiglip-224px+7b",
        "7b-clip"  : "prism-clip+7b",
    },
    display_name = "PrismaticVLM",
    plot_color   = "#e377c2",
)

MODEL_REGISTRY["prismatic_hf"] = ModelSpec(
    # HuggingFace-based loader — use when huggingface_hub >= 0.21 supports '+'.
    extractor_class   = PrismaticExtractor,
    checkpoints       = {
        "7b-dino"  : "TRI-ML/prism-dinosiglip-224px+7b",
        "7b-clip"  : "TRI-ML/prism-clip+7b",
    },
    display_name      = "PrismaticVLM (HF)",
    plot_color        = "#e377c2",
)

# ── VLM → VLA comparison: Prismatic → OpenVLA ─────────────────────────────────

MODEL_REGISTRY["openvla"] = ModelSpec(
    extractor_class = OpenVLAExtractor,
    checkpoints     = {
        # OpenVLA is fine-tuned from prismatic/7b-dino on 970k OXE episodes.
        # Run alongside prismatic/7b-dino then use the merge config below to compare.
        "7b"  : "openvla/openvla-7b",
        # OpenVLA-OFT (parameter-efficient fine-tuned variant):
        # "7b-oft" : "openvla/openvla-oft-droid-7b",
    },
    display_name = "OpenVLA",
    plot_color   = "#d62728",
)

# ── VLM → VLA comparison: PaliGemma → π0 ─────────────────────────────────────

MODEL_REGISTRY["paligemma"] = ModelSpec(
    extractor_class   = PaliGemmaExtractor,
    checkpoints       = {
        # Pretrained PaliGemma 3B — the base VLM used in π0.
        "3b-pt" : "google/paligemma-3b-pt-224",
    },
    display_name      = "PaliGemma",
    base_processor_id = "google/paligemma-3b-pt-224",
    plot_color        = "#1f77b4",
)

MODEL_REGISTRY["pi0"] = ModelSpec(
    extractor_class   = Pi0Extractor,
    checkpoints       = {
        # π0 base policy (PaliGemma fine-tuned on OXE + PI internal data).
        "base"  : "lerobot/pi0_base",
        # π0.5 (extended training):
        # "pi05" : "lerobot/pi05_base",
    },
    display_name      = "π0",
    # Processor loaded from base PaliGemma (π0 does not bundle its own processor)
    base_processor_id = "google/paligemma-3b-pt-224",
    plot_color        = "#9467bd",
)

# ── Merge configs: VLM vs VLA comparison plots ────────────────────────────────
#
# After running inference for each model separately, generate comparison plots with:
#   python probing.py --model_type prismatic_openvla --merge
#   python probing.py --model_type paligemma_pi0 --merge

MERGE_CONFIGS["prismatic_openvla"] = {
    # Before/after VLA fine-tuning: PrismaticVLM → OpenVLA
    "scale_order"   : ["7b-dino", "7b"],
    "scale_sources" : {"7b-dino": "prismatic", "7b": "openvla"},
}

MERGE_CONFIGS["paligemma_pi0"] = {
    # Before/after VLA fine-tuning: PaliGemma → π0
    "scale_order"   : ["3b-pt", "base"],
    "scale_sources" : {"3b-pt": "paligemma", "base": "pi0"},
}

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


# ── Answer matching ───────────────────────────────────────────────────────────

def check_answer(generated_text: str, expected_category: str, mcq_map: Optional[dict] = None) -> bool:
    if not generated_text or not generated_text.strip():
        return False
    text     = generated_text.strip().lower()
    expected = expected_category.lower()
    opposite = OPPOSITE_MAP[expected]

    if mcq_map:
        exp_letter = mcq_map.get(expected)
        opp_letter = mcq_map.get(opposite)
        if exp_letter and text in (exp_letter, exp_letter+'.', exp_letter+')', exp_letter+','):
            return True
        if opp_letter and text in (opp_letter, opp_letter+'.', opp_letter+')', opp_letter+','):
            return False
    else:
        exp_letter = opp_letter = None

    mcq_exp = f'({exp_letter})' if exp_letter else None
    mcq_opp = f'({opp_letter})' if opp_letter else None

    def earliest(word, mcq_pat=None):
        positions = []
        pos = text.find(word)
        if pos != -1:
            positions.append(pos)
        for syn in SYNONYMS.get(word, []):
            pos = text.find(syn)
            if pos != -1:
                positions.append(pos)
        if mcq_pat:
            pos = text.find(mcq_pat)
            if pos != -1:
                positions.append(pos)
        return min(positions) if positions else -1

    pos_exp = earliest(expected, mcq_exp)
    pos_opp = earliest(opposite, mcq_opp)
    if pos_exp == -1:
        return False
    if pos_opp == -1:
        return True
    return pos_exp < pos_opp


# ── Swap pair creation ────────────────────────────────────────────────────────

def load_swap_pairs(
    tsv_path: str,
    seed: int = 42,
    filter_unknown: bool = True,
    question_type: str = 'short_answer',
) -> List[dict]:
    """Load EmbSpatialBench TSV and create minimal-pair swap pairs.

    Parameters
    ----------
    filter_unknown
        If True (default), skip distance pairs whose target object is unknown/empty
        and drop unknown candidates from the reference object pool.
    question_type
        ``'short_answer'`` uses "Answer with only one word." templates;
        ``'mcq'`` uses A/B multiple-choice templates.
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

                if question_type == 'short_answer':
                    tmpl = SHORT_TEMPLATES[grp]
                    pair = {
                        'index': row['index'], 'question_id': str(row['question_id']),
                        'image_base64': row['image'],
                        'original_question': tmpl.format(obj1=obj1, obj2=obj2),
                        'swapped_question':  tmpl.format(obj1=obj2, obj2=obj1),
                        'original_answer': category,
                        'swapped_answer': SHORT_OPPOSITE_MAP[category],
                        'group': grp, 'category': category,
                        'obj1': obj1, 'obj2': obj2, 'mcq_map': None,
                    }
                else:
                    variant = ('left_first'  if grp == 'horizontal' else 'above_first') \
                              if len(pairs) % 2 == 0 else \
                              ('right_first' if grp == 'horizontal' else 'below_first')
                    tmpl    = MCQ_TEMPLATES[grp][variant]
                    mcq_map = MCQ_LETTER[grp][variant]
                    pair = {
                        'index': row['index'], 'question_id': str(row['question_id']),
                        'image_base64': row['image'],
                        'original_question': tmpl.format(obj1=obj1, obj2=obj2),
                        'swapped_question':  tmpl.format(obj1=obj2, obj2=obj1),
                        'original_answer': category,
                        'swapped_answer': OPPOSITE_MAP[category],
                        'group': grp, 'category': category,
                        'obj1': obj1, 'obj2': obj2, 'mcq_map': mcq_map,
                    }

            elif category in ['far', 'close']:
                answer_key     = row['answer']
                options        = {k: row[k] for k in ['A', 'B', 'C', 'D']}
                target_object  = options[answer_key]
                candidates     = [v for k, v in options.items() if k != answer_key]

                if filter_unknown:
                    if not _valid_obj(target_object):
                        continue
                    candidates = [v for v in candidates if _valid_obj(v)]
                    if not candidates:
                        continue

                reference_object = rng.choice(candidates)

                if question_type == 'short_answer':
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
                        'mcq_map': None,
                    }
                else:
                    variant = 'far_first' if len(pairs) % 2 == 0 else 'close_first'
                    tmpl    = MCQ_TEMPLATES['distance'][variant]
                    mcq_map = MCQ_LETTER['distance'][variant]
                    pair = {
                        'index': row['index'], 'question_id': str(row['question_id']),
                        'image_base64': row['image'],
                        'original_question': tmpl.format(ref=reference_object, subj=target_object),
                        'swapped_question':  tmpl.format(ref=target_object,  subj=reference_object),
                        'original_answer': category,
                        'swapped_answer': OPPOSITE_MAP[category],
                        'group': 'distance', 'category': category,
                        'target_object': target_object, 'reference_object': reference_object,
                        'mcq_map': mcq_map,
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


# ── Cross-group quads (distance × vertical) ───────────────────────────────────

def build_hf_bbox_cache(hf_dataset_name: str = 'FlagEval/EmbSpatial-Bench') -> Dict[str, dict]:
    from datasets import load_dataset
    logger.info(f"Loading HF dataset: {hf_dataset_name}")
    ds = load_dataset(hf_dataset_name, split='test')
    cache = {}
    for item in ds:
        qid = str(item['question_id'])
        cache[qid] = {
            'objects':     item['objects'],
            'relation':    item['relation'],
            'data_source': item['data_source'],
            'answer':      item['answer'],
        }
    logger.info(f"Built bbox cache: {len(cache)} entries")
    return cache


def create_cross_group_quads(
    swap_pairs: List[dict],
    hf_cache: Dict[str, dict],
    threshold_ratio: float = 0.05,
    question_type: str = 'short_answer',
) -> List[dict]:
    """For far/close swap pairs, create additional vertical queries using bbox."""
    IMAGE_HEIGHTS = {'ai2thor': 300, 'mp3d': 480, 'scannet': 968}
    quads  = []
    stats  = {'total': 0, 'matched': 0, 'ambiguous': 0, 'no_bbox': 0}

    for pair in [p for p in swap_pairs if p['group'] == 'distance']:
        stats['total'] += 1
        qid = pair['question_id']
        if qid not in hf_cache:
            stats['no_bbox'] += 1
            continue

        hf_item = hf_cache[qid]
        names   = hf_item['objects']['name']
        bboxes  = hf_item['objects']['bbox']
        target  = pair['target_object']
        ref     = pair['reference_object']

        ty = ry = None
        for name, bbox in zip(names, bboxes):
            cy = bbox[1] + bbox[3] / 2
            if name == target:
                ty = cy
            if name == ref:
                ry = cy

        if ty is None or ry is None:
            stats['no_bbox'] += 1
            continue

        image_height = IMAGE_HEIGHTS.get(hf_item['data_source'], 480)
        if abs(ty - ry) < image_height * threshold_ratio:
            stats['ambiguous'] += 1
            continue

        vert_ans = 'above' if ty < ry else 'below'

        if question_type == 'short_answer':
            vtmpl       = SHORT_TEMPLATES['vertical']
            vmcq_map    = None
            vorig_q     = vtmpl.format(obj1=target, obj2=ref)
            vswap_q     = vtmpl.format(obj1=ref,    obj2=target)
            vswap_ans   = SHORT_OPPOSITE_MAP[vert_ans]
        else:
            vvariant    = 'above_first' if len(quads) % 2 == 0 else 'below_first'
            vtmpl       = MCQ_TEMPLATES['vertical'][vvariant]
            vmcq_map    = MCQ_LETTER['vertical'][vvariant]
            vorig_q     = vtmpl.format(obj1=target, obj2=ref)
            vswap_q     = vtmpl.format(obj1=ref,    obj2=target)
            vswap_ans   = OPPOSITE_MAP[vert_ans]

        quads.append({
            'index':              pair['index'],
            'image_base64':       pair['image_base64'],
            'dist_original_q':    pair['original_question'],
            'dist_swapped_q':     pair['swapped_question'],
            'dist_original_answer': pair['original_answer'],
            'dist_swapped_answer':  pair['swapped_answer'],
            'dist_mcq_map':       pair['mcq_map'],
            'vert_original_q':    vorig_q,
            'vert_swapped_q':     vswap_q,
            'vert_original_answer': vert_ans,
            'vert_swapped_answer':  vswap_ans,
            'vert_mcq_map':       vmcq_map,
            'data_source':        hf_item['data_source'],
        })
        stats['matched'] += 1

    logger.info(f"Cross-group quads: {stats['matched']}/{stats['total']} "
                f"(ambiguous={stats['ambiguous']}, no_bbox={stats['no_bbox']})")
    return quads


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

            correct_o  = check_answer(p_o, pair['original_answer'], pair['mcq_map'])
            correct_s  = check_answer(p_s, pair['swapped_answer'],  pair['mcq_map'])

            delta = {
                l: hs_s[l] - hs_o[l]
                for l in extractor.target_layers
                if l in hs_o and l in hs_s
            }

            records.append({
                'index': pair['index'], 'group': pair['group'], 'category': pair['category'],
                'original_answer': pair['original_answer'], 'swapped_answer': pair['swapped_answer'],
                'pred_orig': p_o, 'pred_swap': p_s,
                'is_correct_orig': correct_o, 'is_correct_swap': correct_s,
                'hs_orig': hs_o, 'hs_swap': hs_s, 'delta': delta,
            })

            mark_o = "O" if correct_o else "X"
            mark_s = "O" if correct_s else "X"
            logger.info(f"  #{pair['index']:<6} {pair['category']:<6} "
                        f"orig[{mark_o}]=\"{p_o[:40]}\" swap[{mark_s}]=\"{p_s[:40]}\"")

        except Exception as e:
            logger.warning(f"Error on index {pair['index']}: {e}")

    logger.info(f"Extracted {len(records)} records")
    for cat in CATEGORY_ORDER:
        cat_recs = [r for r in records if r['category'] == cat]
        n = len(cat_recs)
        if not n:
            continue
        c_o = sum(r['is_correct_orig'] for r in cat_recs)
        c_s = sum(r['is_correct_swap'] for r in cat_recs)
        c_b = sum(r['is_correct_orig'] and r['is_correct_swap'] for r in cat_recs)
        logger.info(f"  {cat:>6s} (n={n}): orig={c_o/n:.1%}, swap={c_s/n:.1%}, both={c_b/n:.1%}")
    return records


def extract_cross_group_features(
    extractor: BaseHiddenStateExtractor,
    quads: List[dict],
) -> List[dict]:
    """Extract hidden states for cross-group quads (4 forward passes each)."""
    records = []
    for quad in tqdm(quads, desc="Cross-group quads"):
        try:
            image = decode_base64_image(quad['image_base64'])
            hs_do, pd_o = _run_query(extractor, image, quad['dist_original_q'])
            hs_ds, pd_s = _run_query(extractor, image, quad['dist_swapped_q'])
            hs_vo, pv_o = _run_query(extractor, image, quad['vert_original_q'])
            hs_vs, pv_s = _run_query(extractor, image, quad['vert_swapped_q'])

            delta_dist = {l: hs_ds[l] - hs_do[l] for l in extractor.target_layers if l in hs_do and l in hs_ds}
            delta_vert = {l: hs_vs[l] - hs_vo[l] for l in extractor.target_layers if l in hs_vo and l in hs_vs}

            records.append({
                'index': quad['index'],
                'delta_dist': delta_dist, 'delta_vert': delta_vert,
                'pred_d_orig': pd_o, 'pred_d_swap': pd_s,
                'pred_v_orig': pv_o, 'pred_v_swap': pv_s,
                'is_correct_d_orig': check_answer(pd_o, quad['dist_original_answer'], quad['dist_mcq_map']),
                'is_correct_d_swap': check_answer(pd_s, quad['dist_swapped_answer'],  quad['dist_mcq_map']),
                'is_correct_v_orig': check_answer(pv_o, quad['vert_original_answer'], quad['vert_mcq_map']),
                'is_correct_v_swap': check_answer(pv_s, quad['vert_swapped_answer'],  quad['vert_mcq_map']),
                'data_source': quad['data_source'],
            })
        except Exception as e:
            logger.warning(f"Error on cross-group index {quad['index']}: {e}")

    logger.info(f"Extracted {len(records)} cross-group records")
    return records


# ============================================================================
# Analysis Functions
# ============================================================================

def compute_delta_consistency(records: List[dict], target_layers: List[int]):
    """Compute within-category delta consistency and axis coherence."""
    within_cat_results    = {}
    axis_coherence_results = {}

    for group in GROUP_ORDER:
        canonical = CANONICAL_CATEGORIES[group]
        opposite  = OPPOSITE_MAP[canonical]
        group_recs = [r for r in records if r['group'] == group]

        for layer in target_layers:
            for cat in [canonical, opposite]:
                deltas = [r['delta'][layer] for r in group_recs
                          if r['category'] == cat and layer in r['delta']]
                if len(deltas) >= 2:
                    arr = np.array(deltas)
                    sim = cosine_similarity(arr)
                    upper = sim[np.triu_indices(len(deltas), k=1)]
                    within_cat_results[(cat, layer)] = {
                        'mean': float(np.mean(upper)),
                        'std':  float(np.std(upper)),
                        'n':    len(deltas),
                    }

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
                axis_coherence_results[(group, layer)] = {
                    'mean': float(np.mean(upper)),
                    'std':  float(np.std(upper)),
                    'n':    len(all_deltas),
                }

    return within_cat_results, axis_coherence_results


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


def compute_delta_norm_per_category(records: List[dict], layer: int) -> Optional[pd.DataFrame]:
    """Mean L2 norm of delta vectors per category at a given layer."""
    rows = {}
    for cat in CATEGORY_ORDER:
        deltas = [r['delta'][layer] for r in records if r['category'] == cat and layer in r['delta']]
        if deltas:
            rows[cat] = float(np.mean([np.linalg.norm(d) for d in deltas]))
    if not rows:
        return None
    df = pd.DataFrame.from_dict(rows, orient='index', columns=['norm'])
    return df.loc[[c for c in CATEGORY_ORDER if c in df.index]]


def filter_both_correct(records: List[dict]) -> List[dict]:
    return [r for r in records if r['is_correct_orig'] and r['is_correct_swap']]


def check_category_validity(records: List[dict], scale: str) -> Dict[str, dict]:
    validity = {}
    for cat in CATEGORY_ORDER:
        cat_recs = [r for r in records if r['category'] == cat]
        n = len(cat_recs)
        if n == 0:
            validity[cat] = {'n': 0, 'acc_orig': 0, 'acc_swap': 0, 'reliable': False}
            continue
        acc_orig = sum(r['is_correct_orig'] for r in cat_recs) / n
        acc_swap = sum(r['is_correct_swap'] for r in cat_recs) / n
        reliable = acc_orig >= 0.5 and acc_swap >= 0.5
        validity[cat] = {'n': n, 'acc_orig': acc_orig, 'acc_swap': acc_swap, 'reliable': reliable}
        if not reliable:
            logger.warning(f"  [!] Category '{cat}' unreliable at scale={scale}: "
                           f"orig={acc_orig:.1%}, swap={acc_swap:.1%}")
    return validity


def compute_cross_group_alignment(quad_records: List[dict], target_layers: List[int]) -> dict:
    results = {}
    for layer in target_layers:
        per_sample, delta_verts, delta_dists = [], [], []
        for rec in quad_records:
            if layer in rec['delta_vert'] and layer in rec['delta_dist']:
                dv, dd = rec['delta_vert'][layer], rec['delta_dist'][layer]
                nv, nd = np.linalg.norm(dv), np.linalg.norm(dd)
                if nv > 1e-10 and nd > 1e-10:
                    per_sample.append(float(np.dot(dv, dd) / (nv * nd)))
                    delta_verts.append(dv)
                    delta_dists.append(dd)

        if not per_sample:
            continue

        mean_dv   = np.mean(delta_verts, axis=0)
        mean_dd   = np.mean(delta_dists, axis=0)
        nv, nd    = np.linalg.norm(mean_dv), np.linalg.norm(mean_dd)
        mean_align = float(np.dot(mean_dv, mean_dd) / (nv * nd + 1e-10))

        rng = np.random.RandomState(42)
        perm_aligns = []
        for _ in range(100):
            shuf = [delta_dists[i] for i in rng.permutation(len(delta_dists))]
            pcos = [np.dot(dv, dd) / (np.linalg.norm(dv) * np.linalg.norm(dd))
                    for dv, dd in zip(delta_verts, shuf)
                    if np.linalg.norm(dv) > 1e-10 and np.linalg.norm(dd) > 1e-10]
            perm_aligns.append(np.mean(pcos) if pcos else 0.0)

        results[layer] = {
            'per_sample_mean':    float(np.mean(per_sample)),
            'per_sample_std':     float(np.std(per_sample)),
            'mean_delta_alignment': mean_align,
            'permutation_mean':   float(np.mean(perm_aligns)),
            'permutation_std':    float(np.std(perm_aligns)),
            'n_samples':          len(per_sample),
        }
    return results


def compute_prediction_stats(records: List[dict], scale: str) -> dict:
    stats = {'scale': scale}
    tot_o = tot_s = tot_b = tot_n = 0
    for group in GROUP_ORDER:
        grecs = [r for r in records if r['group'] == group]
        n  = len(grecs)
        co = sum(r['is_correct_orig'] for r in grecs)
        cs = sum(r['is_correct_swap'] for r in grecs)
        cb = sum(r['is_correct_orig'] and r['is_correct_swap'] for r in grecs)
        stats[f'{group}_n']        = n
        stats[f'{group}_acc_orig'] = co / n if n else 0
        stats[f'{group}_acc_swap'] = cs / n if n else 0
        stats[f'{group}_acc_both'] = cb / n if n else 0
        tot_o += co; tot_s += cs; tot_b += cb; tot_n += n
    stats['overall_acc_orig'] = tot_o / tot_n if tot_n else 0
    stats['overall_acc_swap'] = tot_s / tot_n if tot_n else 0
    stats['overall_acc_both'] = tot_b / tot_n if tot_n else 0
    stats['overall_n'] = tot_n
    return stats


# ============================================================================
# Saving & Loading
# ============================================================================

def save_scale_results(
    scale, swap_records, quad_records,
    within_cat_consistency, axis_coherence,
    cross_alignment, pred_stats, target_layers,
    category_validity, delta_heatmaps,
    output_dir, both_correct_tag='all_pairs',
    save_alignment=True, delta_norms=None,
):
    csv_dir  = os.path.join(output_dir, 'csv')
    json_dir = os.path.join(output_dir, 'json')
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    pred_rows = [{
        'index': r['index'], 'group': r['group'], 'category': r['category'],
        'pred_orig': r['pred_orig'], 'pred_swap': r['pred_swap'],
        'is_correct_orig': r['is_correct_orig'], 'is_correct_swap': r['is_correct_swap'],
    } for r in swap_records]
    pd.DataFrame(pred_rows).to_csv(
        os.path.join(csv_dir, f'predictions_{scale}_{both_correct_tag}.csv'), index=False)

    wc_data = {f'{cat}_L{layer}': vals for (cat, layer), vals in within_cat_consistency.items()}
    with open(os.path.join(json_dir, f'within_cat_consistency_{scale}_{both_correct_tag}.json'), 'w') as f:
        json.dump(wc_data, f, indent=2)

    coh_data = {f'{group}_L{layer}': vals for (group, layer), vals in axis_coherence.items()}
    with open(os.path.join(json_dir, f'axis_coherence_{scale}_{both_correct_tag}.json'), 'w') as f:
        json.dump(coh_data, f, indent=2)

    if save_alignment and cross_alignment:
        with open(os.path.join(json_dir, f'cross_alignment_{scale}.json'), 'w') as f:
            json.dump({f'L{layer}': vals for layer, vals in cross_alignment.items()}, f, indent=2)

    with open(os.path.join(json_dir, f'pred_stats_{scale}.json'), 'w') as f:
        json.dump(pred_stats, f, indent=2)

    with open(os.path.join(json_dir, f'category_validity_{scale}.json'), 'w') as f:
        json.dump(category_validity, f, indent=2)

    for layer, df in delta_heatmaps.items():
        if df is not None:
            df.to_csv(os.path.join(csv_dir, f'delta_similarity_{scale}_L{layer}_{both_correct_tag}.csv'))

    if delta_norms:
        for layer, df in delta_norms.items():
            if df is not None:
                df.to_csv(os.path.join(csv_dir, f'delta_norm_{scale}_L{layer}_{both_correct_tag}.csv'))

    logger.info(f"Saved results for scale={scale} ({both_correct_tag}) to {output_dir}")


def save_vectors_npz(scale, swap_records, target_layers, output_dir):
    delta_data = {}
    for layer in target_layers:
        groups_, cats_, vecs_, corig_, cswap_, idxs_ = [], [], [], [], [], []
        orig_, swap_, lbls_ = [], [], []
        for r in swap_records:
            if layer in r['delta']:
                groups_.append(r['group']); cats_.append(r['category'])
                vecs_.append(r['delta'][layer])
                corig_.append(r['is_correct_orig']); cswap_.append(r['is_correct_swap'])
                idxs_.append(r['index'])
            if layer in r['hs_orig'] and layer in r['hs_swap']:
                orig_.append(r['hs_orig'][layer]); swap_.append(r['hs_swap'][layer])
                lbls_.append(r['category'])
        if vecs_:
            delta_data[f'delta_L{layer}']          = np.array(vecs_)
            delta_data[f'groups_L{layer}']          = np.array(groups_)
            delta_data[f'categories_L{layer}']      = np.array(cats_)
            delta_data[f'is_correct_orig_L{layer}'] = np.array(corig_)
            delta_data[f'is_correct_swap_L{layer}'] = np.array(cswap_)
            delta_data[f'indices_L{layer}']         = np.array(idxs_)
        if orig_:
            delta_data[f'orig_L{layer}']   = np.array(orig_)
            delta_data[f'swap_L{layer}']   = np.array(swap_)
            delta_data[f'labels_L{layer}'] = np.array(lbls_)

    npz_dir = os.path.join(output_dir, 'npz')
    os.makedirs(npz_dir, exist_ok=True)
    np.savez_compressed(os.path.join(npz_dir, f'vectors_{scale}.npz'), **delta_data)
    logger.info(f"Saved vectors NPZ for scale={scale}")


def save_cross_group_npz(scale, quad_records, target_layers, output_dir):
    if not quad_records:
        return
    cg_data = {}
    for layer in target_layers:
        dverts = [r['delta_vert'][layer] for r in quad_records if layer in r['delta_vert']]
        ddists = [r['delta_dist'][layer] for r in quad_records if layer in r['delta_dist']]
        if dverts:
            cg_data[f'delta_vert_L{layer}'] = np.array(dverts)
            cg_data[f'delta_dist_L{layer}'] = np.array(ddists)
    npz_dir = os.path.join(output_dir, 'npz')
    os.makedirs(npz_dir, exist_ok=True)
    np.savez_compressed(os.path.join(npz_dir, f'cross_group_vectors_{scale}.npz'), **cg_data)


def save_cross_alignment(scale, cross_alignment, output_dir):
    json_dir = os.path.join(output_dir, 'json')
    os.makedirs(json_dir, exist_ok=True)
    with open(os.path.join(json_dir, f'cross_alignment_{scale}.json'), 'w') as f:
        json.dump({f'L{layer}': vals for layer, vals in cross_alignment.items()}, f, indent=2)


def load_axis_coherence(output_dir, scale, tag='all_pairs'):
    path = os.path.join(output_dir, 'json', f'axis_coherence_{scale}_{tag}.json')
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = json.load(f)
    return {(p[0], int(p[1])): vals for key, vals in raw.items()
            for p in [key.rsplit('_L', 1)] if len(p) == 2}


def load_within_cat_consistency(output_dir, scale, tag='all_pairs'):
    path = os.path.join(output_dir, 'json', f'within_cat_consistency_{scale}_{tag}.json')
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = json.load(f)
    return {(p[0], int(p[1])): vals for key, vals in raw.items()
            for p in [key.rsplit('_L', 1)] if len(p) == 2}


def load_scale_alignment(output_dir, scale):
    path = os.path.join(output_dir, 'json', f'cross_alignment_{scale}.json')
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = json.load(f)
    return {int(k[1:]): v for k, v in raw.items() if k.startswith('L')}


def load_delta_heatmaps(output_dir, scale, tag='all_pairs'):
    import glob as glob_mod
    pattern = os.path.join(output_dir, 'csv', f'delta_similarity_{scale}_L*_{tag}.csv')
    result  = {}
    for fpath in glob_mod.glob(pattern):
        m = re.search(
            rf'delta_similarity_{re.escape(scale)}_L(\d+)_{re.escape(tag)}\.csv$',
            os.path.basename(fpath))
        if m:
            result[int(m.group(1))] = pd.read_csv(fpath, index_col=0)
    return result


def _model_key(model_type: str, scale: str) -> str:
    return f"{model_type}_{scale}"


def _has_phase_b_data(scale_dir: str, scale: str) -> bool:
    path = os.path.join(scale_dir, 'json', f'cross_alignment_{scale}.json')
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            return bool(json.load(f))
    except Exception:
        return False


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


def plot_within_cat_trajectory(within_cat, scale, model_type, save_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    for cat in CATEGORY_ORDER:
        layers, vals = zip(*sorted(
            ((l, v['mean']) for (c, l), v in within_cat.items() if c == cat),
            key=lambda x: x[0]
        )) if any(c == cat for (c, _) in within_cat) else ([], [])
        if layers:
            ax.plot(layers, vals, '-o', color=CAT_COLORS[cat], label=cat, linewidth=2, markersize=3)
    ax.set_xlabel('Layer Index'); ax.set_ylabel('Within-Category Consistency')
    ax.set_title(f'{model_type} ({scale}) – Within-Category Delta Consistency', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
    logger.info(f"Saved: {save_path}")


def plot_cross_group_alignment_trajectory(cross_alignment, scale, model_type, save_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    layers = sorted(cross_alignment.keys())
    ax.plot(layers, [cross_alignment[l]['per_sample_mean'] for l in layers],
            '-o', color='#d62728', label='cos(d_vert, d_dist) per-sample mean', linewidth=2.5, markersize=3)
    ax.plot(layers, [cross_alignment[l]['mean_delta_alignment'] for l in layers],
            '--s', color='#e377c2', label='cos(mean_d_vert, mean_d_dist)', linewidth=1.5, markersize=3)
    pm = [cross_alignment[l]['permutation_mean'] for l in layers]
    ps = [cross_alignment[l]['permutation_std']  for l in layers]
    ax.plot(layers, pm, ':', color='gray', label='permutation control', linewidth=1.5)
    ax.fill_between(layers, [m - 2*s for m, s in zip(pm, ps)],
                             [m + 2*s for m, s in zip(pm, ps)], alpha=0.2, color='gray')
    ax.set_xlabel('Layer Index'); ax.set_ylabel('Cosine Alignment')
    ax.set_title(f'{model_type} ({scale}) – Cross-Group Alignment', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
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


def plot_pred_stats_trajectory(all_pred_stats, model_type, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    scales = [d['scale'] for d in all_pred_stats]
    for group in GROUP_ORDER:
        y_vals = [d.get(f'{group}_acc_both', 0) for d in all_pred_stats]
        ax.plot(range(len(scales)), y_vals, '-o', color=GROUP_COLORS[group],
                label=group, linewidth=2.5, markersize=6)
    ax.set_xticks(range(len(scales))); ax.set_xticklabels(scales)
    ax.set_xlabel('Scale'); ax.set_ylabel('Accuracy (both correct)')
    ax.set_title(f'{model_type} – Both-Correct Accuracy Across Scales', fontweight='bold')
    ax.legend(fontsize=10); ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
    logger.info(f"Saved: {save_path}")


def plot_pca_embeddings(vectors_npz_path, scale, model_type, save_dir, bc_only=False):
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
        if bc_only and deltas is not None:
            co = data.get(f'is_correct_orig_L{layer}')
            cs = data.get(f'is_correct_swap_L{layer}')
            if co is not None and cs is not None:
                mask = co.astype(bool) & cs.astype(bool)
                if orig is not None and len(orig) == len(mask):
                    orig = orig[mask]; swap = swap[mask]
                    labels = labels[mask] if labels is not None else None
                if len(deltas) == len(mask):
                    deltas = deltas[mask]
                    cats   = cats[mask]   if cats   is not None else None
                    groups = groups[mask] if groups is not None else None
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


def plot_pca_3d(vectors_npz_path, scale, model_type, save_dir, bc_only=False):
    data = np.load(vectors_npz_path, allow_pickle=True)
    layers = sorted(int(k.replace('orig_L', '')) for k in data.files if k.startswith('orig_L'))
    if not layers:
        return
    os.makedirs(save_dir, exist_ok=True)
    for layer in layers:
        deltas = data.get(f'delta_L{layer}')
        cats   = data.get(f'categories_L{layer}')
        if bc_only and deltas is not None:
            co = data.get(f'is_correct_orig_L{layer}')
            cs = data.get(f'is_correct_swap_L{layer}')
            if co is not None and cs is not None:
                mask = co.astype(bool) & cs.astype(bool)
                if len(deltas) == len(mask):
                    deltas = deltas[mask]
                    cats   = cats[mask] if cats is not None else None
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
        for tag, subfolder in [('all_pairs', 'all'), ('both_correct', 'both_correct')]:
            out_dir = os.path.join(model_dir, 'plots', subfolder, 'heatmap')
            os.makedirs(out_dir, exist_ok=True)
            for layer in npz_layers:
                csv_path = os.path.join(csv_dir, f'delta_similarity_{scale}_L{layer}_{tag}.csv')
                if not os.path.exists(csv_path):
                    continue
                df = pd.read_csv(csv_path, index_col=0)
                available = [c for c in CATEGORY_ORDER if c in df.index]
                if not available:
                    continue
                plot_delta_heatmap(
                    df.loc[available, available],
                    f'{model_type} ({scale}) — Delta Heatmap L{layer} '
                    f'({"both-correct" if tag == "both_correct" else "all pairs"})',
                    os.path.join(out_dir, f'heatmap_{scale}_L{layer}.png'),
                )


def run_all_layer_pca(model_dir: str, model_type: str, scales: List[str]):
    for scale in scales:
        npz_path = os.path.join(model_dir, 'npz', f'vectors_{scale}.npz')
        if not os.path.exists(npz_path):
            continue
        for bc_only, subfolder in [(False, 'all'), (True, 'both_correct')]:
            for plot_fn, sub2 in [(plot_pca_embeddings, 'pca'), (plot_pca_3d, 'pca_3d')]:
                out = os.path.join(model_dir, 'plots', subfolder, sub2)
                os.makedirs(out, exist_ok=True)
                plot_fn(npz_path, scale, model_type, out, bc_only=bc_only)


def run_accuracy_charts(pred_stats, cat_validity, model_type, save_dir):
    """Generate accuracy charts into save_dir."""
    os.makedirs(save_dir, exist_ok=True)
    scales = [d['scale'] for d in pred_stats]

    # Group bars
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    x = np.arange(3); width = 0.8 / max(len(scales), 1)
    for idx, group in enumerate(GROUP_ORDER):
        ax = axes[idx]
        for i, scale in enumerate(scales):
            entry = next((d for d in pred_stats if d['scale'] == scale), None)
            if entry is None:
                continue
            vals = [entry.get(f'{group}_acc_orig', 0),
                    entry.get(f'{group}_acc_swap', 0),
                    entry.get(f'{group}_acc_both', 0)]
            offset = (i - len(scales) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=scale,
                   color=SCALE_COLORS.get(scale, _DEFAULT_SCALE_COLOR), alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(['orig', 'swap', 'both'])
        ax.set_ylabel('Accuracy'); ax.set_title(group.capitalize(), fontweight='bold')
        ax.legend(fontsize=7); ax.set_ylim(0, 1.15)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5); ax.grid(True, alpha=0.3, axis='y')
    fig.suptitle(f'{model_type} – Prediction Accuracy by Group', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy_group_bars.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # Both-correct trajectory
    plot_pred_stats_trajectory(pred_stats, model_type,
                               os.path.join(save_dir, 'accuracy_trajectory.png'))
    logger.info(f"Saved accuracy charts to {save_dir}")


# ============================================================================
# Main Pipeline
# ============================================================================

def process_scale(args, model_type: str, scale: str, swap_pairs: List[dict], quads: List[dict]):
    """Run full inference + analysis for one (model_type, scale) checkpoint."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {model_type} / {scale}")
    logger.info(f"{'='*60}")

    extractor    = get_extractor(model_type, scale, device=args.device)
    target_layers = extractor.target_layers

    output_dir = os.path.join(args.output_dir, _model_key(model_type, scale))
    plots_dir  = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # ── Phase A: swap pair feature extraction ─────────────────────────────────
    logger.info("\n--- Phase A: Swap pair extraction ---")
    swap_records = extract_swap_features(
        extractor, swap_pairs, max_samples_per_category=args.max_samples_per_category)

    # ── Phase C_A: analysis ───────────────────────────────────────────────────
    logger.info("\n--- Phase C_A: Analysis ---")
    cat_validity     = check_category_validity(swap_records, scale)
    within_cat_all,  coh_all = compute_delta_consistency(swap_records, target_layers)
    bc_records       = filter_both_correct(swap_records)
    logger.info(f"  Both-correct pairs: {len(bc_records)}/{len(swap_records)}")
    within_cat_bc, coh_bc = compute_delta_consistency(bc_records, target_layers)
    pred_stats = compute_prediction_stats(swap_records, scale)

    dh_all = {}; dn_all = {}; dh_bc = {}; dn_bc = {}
    for layer in target_layers:
        dh_all[layer] = compute_delta_similarity_matrix(swap_records, layer)
        dn_all[layer] = compute_delta_norm_per_category(swap_records, layer)
        if bc_records:
            dh_bc[layer] = compute_delta_similarity_matrix(bc_records, layer)
            dn_bc[layer] = compute_delta_norm_per_category(bc_records, layer)

    max_layer = max(target_layers)
    for group in GROUP_ORDER:
        if (group, max_layer) in coh_all:
            v = coh_all[(group, max_layer)]
            logger.info(f"  axis coherence [{group}, L{max_layer}]: {v['mean']:.4f} ± {v['std']:.4f}")
    logger.info(f"  Accuracy orig={pred_stats['overall_acc_orig']:.1%}, "
                f"swap={pred_stats['overall_acc_swap']:.1%}, "
                f"both={pred_stats['overall_acc_both']:.1%}")

    # ── Phase D_A: save Phase A results ──────────────────────────────────────
    logger.info("\n--- Phase D_A: Saving Phase A results ---")
    save_vectors_npz(scale, swap_records, target_layers, output_dir)
    for tag, wc, sc, dh, dn, recs in [
        ('all_pairs',    within_cat_all, coh_all, dh_all, dn_all, swap_records),
        ('both_correct', within_cat_bc,  coh_bc,  dh_bc,  dn_bc,  bc_records),
    ]:
        if tag == 'both_correct' and not bc_records:
            continue
        save_scale_results(scale, recs, [], wc, sc, {}, pred_stats, target_layers,
                           cat_validity, dh, output_dir, both_correct_tag=tag,
                           save_alignment=False, delta_norms=dn)

    # ── Phase E_A: per-scale plots ────────────────────────────────────────────
    if not args.phase1_only:
        logger.info("\n--- Phase E_A: Per-scale plots ---")
        for condition, wc_data, coh_data in [
            ('all',          within_cat_all, coh_all),
            ('both_correct', within_cat_bc,  coh_bc),
        ]:
            if condition == 'both_correct' and not bc_records:
                continue
            cond_dir = os.path.join(plots_dir, condition)
            wc_dir = os.path.join(cond_dir, 'within_cat_consistency')
            coh_dir = os.path.join(cond_dir, 'axis_coherence')
            os.makedirs(wc_dir, exist_ok=True); os.makedirs(coh_dir, exist_ok=True)
            plot_within_cat_trajectory(
                wc_data, scale, model_type,
                os.path.join(wc_dir, f'within_cat_{scale}.png'))
            plot_axis_coherence_trajectory(
                coh_data, scale, model_type,
                os.path.join(coh_dir, f'axis_coherence_{scale}.png'))

        npz_path = os.path.join(output_dir, 'npz', f'vectors_{scale}.npz')
        if os.path.exists(npz_path):
            run_all_layer_pca(output_dir, model_type, [scale])
        run_all_layer_heatmaps(output_dir, model_type, [scale])
        run_accuracy_charts([pred_stats], cat_validity, model_type,
                            os.path.join(output_dir, 'accuracy'))

    # ── Phase B: cross-group extraction ───────────────────────────────────────
    skip_b = getattr(args, 'skip_phase_b', True) or getattr(args, 'skip_cross_group', True)
    if skip_b or not quads:
        quad_records = []; cross_alignment = {}
    else:
        logger.info("\n--- Phase B: Cross-group extraction ---")
        quad_records    = extract_cross_group_features(extractor, quads)
        cross_alignment = compute_cross_group_alignment(quad_records, target_layers)
        if max_layer in cross_alignment:
            ca = cross_alignment[max_layer]
            logger.info(f"  Cross-group alignment L{max_layer}: "
                        f"{ca['per_sample_mean']:.4f} (perm={ca['permutation_mean']:.4f})")
        save_cross_group_npz(scale, quad_records, target_layers, output_dir)
        save_cross_alignment(scale, cross_alignment, output_dir)

        if not args.phase1_only:
            for condition in ['all', 'both_correct']:
                if condition == 'both_correct' and not bc_records:
                    continue
                ca_dir = os.path.join(plots_dir, condition, 'cross_alignment')
                os.makedirs(ca_dir, exist_ok=True)
                plot_cross_group_alignment_trajectory(
                    cross_alignment, scale, model_type,
                    os.path.join(ca_dir, f'cross_alignment_{scale}.png'))

    del swap_records, bc_records, quad_records
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

    all_coh = {}; all_coh_bc = {}; all_wc = {}; all_wc_bc = {}
    all_align = {}; all_ps = []; all_cv = {}; all_dh = {}; all_dh_bc = {}

    for scale in available_scales:
        sd = _sd(scale)
        coh   = load_axis_coherence(sd, scale, 'all_pairs')
        coh_bc = load_axis_coherence(sd, scale, 'both_correct')
        wc    = load_within_cat_consistency(sd, scale, 'all_pairs')
        wc_bc = load_within_cat_consistency(sd, scale, 'both_correct')
        align = load_scale_alignment(sd, scale)
        dh    = load_delta_heatmaps(sd, scale, 'all_pairs')
        dh_bc = load_delta_heatmaps(sd, scale, 'both_correct')
        pred_path = os.path.join(sd, 'json', f'pred_stats_{scale}.json')
        cv_path   = os.path.join(sd, 'json', f'category_validity_{scale}.json')
        if coh:    all_coh[scale]    = coh
        if coh_bc: all_coh_bc[scale] = coh_bc
        if wc:    all_wc[scale]    = wc
        if wc_bc: all_wc_bc[scale] = wc_bc
        if align: all_align[scale] = align
        if dh:    all_dh[scale]    = dh
        if dh_bc: all_dh_bc[scale] = dh_bc
        if os.path.exists(pred_path):
            with open(pred_path) as f:
                all_ps.append(json.load(f))
        if os.path.exists(cv_path):
            with open(cv_path) as f:
                all_cv[scale] = json.load(f)
        logger.info(f"  Loaded data for '{scale}'")

    if not args.phase1_only:
        for condition, coh_data, wc_data, dh_data, tag in [
            ('all',          all_coh,    all_wc,    all_dh,    'all pairs'),
            ('both_correct', all_coh_bc, all_wc_bc, all_dh_bc, 'both-correct'),
        ]:
            cond_dir = os.path.join(plots_dir, condition)
            os.makedirs(os.path.join(cond_dir, 'axis_coherence'), exist_ok=True)
            os.makedirs(os.path.join(cond_dir, 'within_cat_consistency'), exist_ok=True)
            if len(coh_data) > 1:
                plot_cross_scale_axis_coherence(
                    coh_data, model_type,
                    os.path.join(cond_dir, 'axis_coherence', 'cross_scale.png'),
                    title_prefix=f'Axis Coherence ({tag})')
        if all_ps:
            plot_pred_stats_trajectory(
                all_ps, model_type,
                os.path.join(plots_dir, 'accuracy_trajectory.png'))
        if all_ps:
            run_accuracy_charts(all_ps, all_cv, model_type,
                                os.path.join(plots_dir, 'accuracy'))

    # Summary CSV
    if all_ps:
        rows = []
        for scale in available_scales:
            ps = next((p for p in all_ps if p.get('scale') == scale), None)
            if ps:
                row = dict(ps)
                if scale in all_align:
                    ml = max(all_align[scale].keys())
                    row['alignment_deepest'] = all_align[scale][ml]['per_sample_mean']
                rows.append(row)
        if rows:
            csv_dir = os.path.join(merge_out, 'csv')
            os.makedirs(csv_dir, exist_ok=True)
            pd.DataFrame(rows).to_csv(os.path.join(csv_dir, 'summary.csv'), index=False)

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
  python probing.py --model_type prismatic --scales 7b-dino

  # Run all checkpoints of a model family
  python probing.py --model_type qwen25

  # Generate cross-scale comparison plots from saved results
  python probing.py --model_type qwen25 --merge
""")

    parser.add_argument('--data_path', type=str,
                        default='/data/shared/Qwen/EmbSpatial-Bench/EmbSpatial-Bench.tsv',
                        help='Path to the EmbSpatialBench TSV file.')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=all_model_types,
                        help='Model family to run (see registered models above).')
    parser.add_argument('--scales', type=str, nargs='+', default=None,
                        help='Scale names to process (default: all for the given model_type).')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Root directory for saved results. '
                             'Defaults to <script_dir>/<question_type>/saved_data.')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--merge', action='store_true',
                        help='Merge mode: generate cross-scale plots from saved data.')
    parser.add_argument('--merge-output-dir', type=str, default=None, dest='merge_output_dir',
                        help='Override the output directory for cross-scale plots.')
    parser.add_argument('--group-name', type=str, default=None, dest='group_name',
                        help='Folder name under compare/ for merged output.')
    parser.add_argument('--skip-cross-group', action='store_true', default=True,
                        help='Skip Phase B (cross-group quad extraction). Default: True.')
    parser.add_argument('--run-phase-b', action='store_false', dest='skip_cross_group',
                        help='Enable Phase B (cross-group quad extraction).')
    parser.add_argument('--skip-phase-b', action='store_true', dest='skip_phase_b',
                        help='Alias for --skip-cross-group.')
    parser.add_argument('--max-samples-per-category', type=int, default=200,
                        dest='max_samples_per_category',
                        help='Maximum swap pairs per spatial category (0 = no limit).')
    parser.add_argument('--no-filtering', action='store_true', dest='no_filtering',
                        help='Disable Unknown/empty filtering for far/close pairs.')
    parser.add_argument('--question-type', type=str, default='short_answer',
                        choices=['short_answer', 'mcq'], dest='question_type',
                        help='Prompt format: short_answer (default) or mcq.')
    parser.add_argument('--phase1-only', action='store_true', dest='phase1_only',
                        help='Skip all plot generation; only save data (npz/csv/json).')

    args = parser.parse_args()

    _folder = 'results' if args.question_type == 'short_answer' else 'results_mcq'
    if args.output_dir is None:
        args.output_dir = os.path.join(_HERE, _folder, 'saved_data')
    log_dir = os.path.join(_HERE, _folder, 'logs')

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

    # ── Inference mode ────────────────────────────────────────────────────────
    logger.info("\n=== Loading swap pairs ===")
    swap_pairs = load_swap_pairs(
        args.data_path, args.seed,
        filter_unknown=not args.no_filtering,
        question_type=args.question_type,
    )

    quads = []
    if not args.skip_cross_group and not args.skip_phase_b:
        try:
            hf_cache = build_hf_bbox_cache()
            quads    = create_cross_group_quads(swap_pairs, hf_cache,
                                                question_type=args.question_type)
        except Exception as e:
            logger.warning(f"Cross-group setup failed: {e}. Skipping.")

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
            process_scale(args, args.model_type, scale, swap_pairs, quads)
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
