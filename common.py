"""Shared helpers for every stage: config, dataset, prompt, model loading, residual capture."""

import contextlib
import json
import os

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.abspath(__file__))


def resolve(path):
    """Resolve a config path relative to the project root."""
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def load_config(path=None):
    with open(resolve(path or "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_queries(cfg, num_queries=None):
    """Return the first N queries of the configured split, with their reference answers."""
    with open(resolve(cfg["data"]["dataset"]), encoding="utf-8") as f:
        data = json.load(f)
    split = cfg["data"]["split"]
    n = num_queries or cfg["data"]["num_queries"]
    return [
        {
            "pair_id": pair["pair_id"],
            "category": pair["category"],
            "difficulty": pair["difficulty"],
            "query": pair[split]["query"],
            "reference_answer": pair[split]["final_answer"],
            "reference_reasoning": pair[split]["reasoning"],
        }
        for pair in data["pairs"][:n]
    ]


def build_prompt(tokenizer, query, cfg):
    """Chat-templated prompt ending in '<|im_start|>assistant\\n', where reasoning begins."""
    messages = [
        {"role": "system", "content": cfg["prompt"]["system"]},
        {"role": "user", "content": cfg["prompt"]["instruction"] + query},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_model_and_tokenizer(cfg, device="cuda", attn_implementation=None):
    """attn_implementation="eager" is needed wherever we differentiate through the model
    (forward-mode AD is not supported by the fused SDPA kernels)."""
    name = cfg["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(name)
    kwargs = {"attn_implementation": attn_implementation} if attn_implementation else {}
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=getattr(torch, cfg["model"]["dtype"]), device_map=device, **kwargs
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


def layer_dir(root, kind, layer):
    return os.path.join(root, kind, f"layer_{layer:02d}")


class ResidualRecorder:
    """Captures the raw residual stream of the first forward call inside `capture()`.

    Index 0 is the embedding output (input to block 0); index k (1..L) is the output of
    decoder block k-1. These are raw residual states: the final RMSNorm is not applied,
    unlike the last entry of HF `output_hidden_states`.

    Only the first forward call is recorded, so during `generate()` this is the prefill
    over the whole prompt; decode steps are ignored.
    """

    def __init__(self, model):
        self.num_layers = len(model.model.layers)
        self.enabled = False
        self.store = {}
        self.handles = [model.model.embed_tokens.register_forward_hook(self._hook(0))]
        for i, block in enumerate(model.model.layers):
            self.handles.append(block.register_forward_hook(self._hook(i + 1)))

    def _hook(self, idx):
        def hook(_module, _inp, out):
            if not self.enabled or idx in self.store:
                return
            hs = out[0] if isinstance(out, tuple) else out
            self.store[idx] = hs[0].detach()  # [seq, d]
        return hook

    @contextlib.contextmanager
    def capture(self):
        self.store = {}
        self.enabled = True
        try:
            yield self
        finally:
            self.enabled = False
        missing = self.num_layers + 1 - len(self.store)
        if missing:
            raise RuntimeError(f"ResidualRecorder missed {missing} layers")

    def layers(self):
        """List of [seq, d] tensors, index 0..L."""
        return [self.store[i] for i in range(self.num_layers + 1)]

    def remove(self):
        for h in self.handles:
            h.remove()
