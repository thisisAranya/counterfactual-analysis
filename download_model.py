"""Check whether the model is already in the Hugging Face cache; download it only if missing.

Usage:
    python download_model.py --model Qwen/Qwen2.5-7B-Instruct
Prints the local snapshot path on success. Uses HF_HOME for the cache location.
"""

import argparse
import json
import os

from huggingface_hub import snapshot_download

ALLOW = ["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken", "merges.txt"]


def is_complete(path):
    """True if config, tokenizer and every weight shard are present."""
    if not os.path.isfile(os.path.join(path, "config.json")):
        return False
    if not any(os.path.isfile(os.path.join(path, f))
               for f in ("tokenizer.json", "tokenizer_config.json")):
        return False
    index = os.path.join(path, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index, encoding="utf-8") as f:
            shards = set(json.load(f)["weight_map"].values())
        return all(os.path.isfile(os.path.join(path, s)) for s in shards)
    return os.path.isfile(os.path.join(path, "model.safetensors"))


def ensure_model(model_id, revision=None):
    try:
        path = snapshot_download(model_id, revision=revision, local_files_only=True,
                                 allow_patterns=ALLOW)
        if is_complete(path):
            print(f"Model found in cache: {path}")
            return path
        print(f"Cached copy at {path} is incomplete; downloading missing files...")
    except Exception:
        print(f"Model {model_id} not in cache; downloading...")

    path = snapshot_download(model_id, revision=revision, allow_patterns=ALLOW)
    if not is_complete(path):
        raise RuntimeError(f"Download finished but files are missing in {path}")
    print(f"Model downloaded: {path}")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--revision", default=None)
    args = p.parse_args()
    ensure_model(args.model, args.revision)


if __name__ == "__main__":
    main()
