"""Check whether the model is already in the Hugging Face cache; download it only if missing.

Usage:
    python download_model.py --model Qwen/Qwen2.5-7B-Instruct
Prints the local snapshot path on success. Uses HF_HOME for the cache location.

Downloads use plain HTTPS by default: the Xet transfer backend has failed on Zaratan
("CAS Client Error"). Pass --use_xet to allow it. Interrupted downloads resume, and
failed attempts are retried with a growing wait.
"""

import argparse
import json
import os
import sys
import time

# Must be set before huggingface_hub is imported, which reads it at import time.
if "--use_xet" not in sys.argv:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import snapshot_download  # noqa: E402

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


def download_with_retries(model_id, revision, retries, wait):
    for attempt in range(1, retries + 1):
        try:
            return snapshot_download(model_id, revision=revision, allow_patterns=ALLOW)
        except Exception as e:
            if attempt == retries:
                raise
            print(f"Attempt {attempt}/{retries} failed: {type(e).__name__}: {e}")
            print(f"Retrying in {wait * attempt}s (completed files are kept)...")
            time.sleep(wait * attempt)


def ensure_model(model_id, revision=None, retries=3, wait=30):
    try:
        path = snapshot_download(model_id, revision=revision, local_files_only=True,
                                 allow_patterns=ALLOW)
        if is_complete(path):
            print(f"Model found in cache: {path}")
            return path
        print(f"Cached copy at {path} is incomplete; downloading missing files...")
    except Exception:
        print(f"Model {model_id} not in cache; downloading...")

    print(f"HF_HOME={os.environ.get('HF_HOME', '(default ~/.cache/huggingface)')}, "
          f"xet={'off' if os.environ.get('HF_HUB_DISABLE_XET') == '1' else 'on'}")
    path = download_with_retries(model_id, revision, retries, wait)
    if not is_complete(path):
        raise RuntimeError(f"Download finished but files are missing in {path}")
    print(f"Model downloaded: {path}")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--revision", default=None)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--wait", type=int, default=30, help="seconds; grows with each attempt")
    p.add_argument("--use_xet", action="store_true", help="allow the Xet transfer backend")
    args = p.parse_args()
    ensure_model(args.model, args.revision, args.retries, args.wait)


if __name__ == "__main__":
    main()
