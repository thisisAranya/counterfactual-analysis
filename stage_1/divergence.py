"""
Choose the soft-rollout horizon K from real model behaviour.

For each pair, compare the greedy reasoning the model generated for the original and
the counterfactual query (both from run_stage1.py) and find the first generated token
where they differ. K must reach past that point, otherwise the stage-2 rollout only
sees reasoning that is identical in both versions.

Suggested K = (percentile of first-divergence positions) + margin, rounded up to a
multiple of 8. Settings live in config.yaml under stage_1.divergence.

Run from the project root, after both splits are done:
    python stage_1/divergence.py
Writes <output_root>/stage_1/divergence.json.
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load_config, resolve  # noqa: E402

CONTEXT_TOKENS = 12


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def percentile(values, q):
    """Nearest-rank percentile."""
    s = sorted(values)
    k = max(0, math.ceil(q / 100 * len(s)) - 1)
    return s[k]


def load_baseline(root, split):
    path = os.path.join(root, split, "baseline.json")
    if not os.path.isfile(path):
        sys.exit(f"Missing {path}; run: python stage_1/run_stage1.py --split {split}")
    with open(path, encoding="utf-8") as f:
        return {e["pair_id"]: e for e in json.load(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dcfg = cfg["stage_1"]["divergence"]
    root = os.path.join(resolve(cfg["paths"]["output_root"]), "stage_1")

    orig = load_baseline(root, "original")
    cf = load_baseline(root, "counterfactual")
    pair_ids = [p for p in orig if p in cf]
    if not pair_ids:
        sys.exit("No pair_ids shared between the two splits")

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
        decode = lambda ids: tok.decode(ids, skip_special_tokens=True)  # noqa: E731
    except Exception as e:  # context text is a convenience, not required
        print(f"(tokenizer unavailable, skipping context text: {e})")
        decode = None

    rows = []
    for pid in pair_ids:
        a, b = orig[pid]["generated_ids"], cf[pid]["generated_ids"]
        k = first_divergence(a, b)
        row = {
            "pair_id": pid,
            "first_divergence": k,
            "len_original": len(a),
            "len_counterfactual": len(b),
            "fraction_of_original": round(k / max(len(a), 1), 3),
        }
        if decode:
            lo = max(0, k - CONTEXT_TOKENS)
            row["shared_prefix_tail"] = decode(a[lo:k])
            row["original_continues"] = decode(a[k:k + CONTEXT_TOKENS])
            row["counterfactual_continues"] = decode(b[k:k + CONTEXT_TOKENS])
        rows.append(row)

    ks = [r["first_divergence"] for r in rows]
    p = percentile(ks, dcfg["percentile"])
    suggested_k = int(math.ceil((p + dcfg["margin"]) / 8) * 8)
    summary = {
        "num_pairs": len(rows),
        "min": min(ks),
        "median": percentile(ks, 50),
        "max": max(ks),
        f"p{dcfg['percentile']}": p,
        "margin": dcfg["margin"],
        "suggested_K": suggested_k,
    }

    print(f"{'pair_id':<16}{'diverge@':>9}{'len_o':>7}{'len_cf':>7}")
    for r in rows:
        print(f"{r['pair_id']:<16}{r['first_divergence']:>9}"
              f"{r['len_original']:>7}{r['len_counterfactual']:>7}")
        if decode:
            print(f"    ...{r['shared_prefix_tail']!r}")
            print(f"    O: {r['original_continues']!r}")
            print(f"    C: {r['counterfactual_continues']!r}")
    print("\n=== Divergence summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    out = os.path.join(root, "divergence.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "pairs": rows}, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out}. Set stage_2.horizon_K in config.yaml from suggested_K.")


if __name__ == "__main__":
    main()
