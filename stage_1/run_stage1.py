"""
Stage 1 -- baseline activations and reasoning (Algorithm 1, steps 1-2).

For each original query Q:
  1. Build the chat prompt; position t is its last token ("\\n" after "<|im_start|>assistant").
  2. Greedy-generate the reasoning R(Q) and parse the answer A(Q). The prefill of this
     generation gives h = H_{l,t}(x) at every layer: the exact state reasoning starts from.
  3. Re-run one teacher-forced forward pass over prompt + generation to get the
     reasoning hidden-state trajectory Z(Q) (used for d_R, PDF Sec. 1.10), and to
     cross-check h.

Outputs (under <output_root>/stage_1/<split>/):
    h/layer_XX/<pair_id>.pt      [d] float32    pre-reasoning activation
    h/layer_XX/all.pt            {"pair_ids", "h": [N, d]}
    Z/layer_XX/<pair_id>.pt      [m, d]         residual state at each generated token
    baseline.json                prompts, token ids, reasoning text, answers, checks
    metadata.json                config snapshot, conventions, summary

Run from the project root:
    python stage_1/run_stage1.py            # all queries
    python stage_1/run_stage1.py --smoke    # 1 query, 64 tokens
    python stage_1/run_stage1.py --split counterfactual   # for divergence.py (choosing K)
"""

import argparse
import copy
import json
import os
import re
import string
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (ResidualRecorder, build_prompt, layer_dir, load_config,  # noqa: E402
                    load_model_and_tokenizer, load_queries, resolve)

ANSWER_RE = re.compile(r"final answer\s*[:：]\s*(.+)", re.IGNORECASE)
ARTICLES = {"a", "an", "the"}
H_COSINE_WARN = 0.999


def parse_answer(text):
    """Last 'Final answer: ...' line, stripped of markdown and trailing punctuation."""
    matches = ANSWER_RE.findall(text)
    if not matches:
        return None
    ans = matches[-1].strip().strip("*").strip()
    return ans.rstrip(".").strip() or None


def normalize(s):
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(w for w in s.split() if w not in ARTICLES)


def auto_match(pred, ref):
    """Lenient match; raw text is saved so every case can be checked by hand."""
    if pred is None:
        return False
    p, r = normalize(pred), normalize(ref)
    if not p:
        return False
    if r in ("yes", "no"):
        return p.split()[0] == r
    return p == r or r in p or p in r


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--smoke", action="store_true", help="1 query, short generation")
    ap.add_argument("--num_queries", type=int, default=None)
    ap.add_argument("--split", choices=["original", "counterfactual"], default=None,
                    help="overrides data.split in the config")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.split:
        cfg["data"]["split"] = args.split
    split = cfg["data"]["split"]
    s1 = cfg["stage_1"]

    num_queries = args.num_queries or cfg["data"]["num_queries"]
    max_new_tokens = s1["max_new_tokens"]
    if args.smoke:
        num_queries = s1["smoke"]["num_queries"]
        max_new_tokens = s1["smoke"]["max_new_tokens"]
    out_root = os.path.join(resolve(cfg["paths"]["output_root"]),
                            "stage_1_smoke" if args.smoke else "stage_1", split)
    traj_dtype = getattr(torch, s1["trajectory_dtype"])

    queries = load_queries(cfg, num_queries)
    model, tokenizer = load_model_and_tokenizer(cfg)
    device = model.device
    rec = ResidualRecorder(model)
    L = rec.num_layers
    d = model.config.hidden_size

    gen_cfg = copy.deepcopy(model.generation_config)
    gen_cfg.do_sample = False
    gen_cfg.temperature = None
    gen_cfg.top_p = None
    gen_cfg.top_k = None
    gen_cfg.repetition_penalty = s1["repetition_penalty"]
    gen_cfg.max_new_tokens = max_new_tokens
    if gen_cfg.pad_token_id is None:
        gen_cfg.pad_token_id = tokenizer.pad_token_id
    eos_ids = gen_cfg.eos_token_id
    eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])

    for kind in ("h", "Z"):
        for i in range(L + 1):
            os.makedirs(layer_dir(out_root, kind, i), exist_ok=True)

    print(f"Model {cfg['model']['name']}: {L} layers, d={d}, device={device}, split={split}")
    print(f"{len(queries)} queries, max_new_tokens={max_new_tokens}, out={out_root}")

    h_all = {i: [] for i in range(L + 1)}
    entries = []
    t0 = time.time()

    for q in queries:
        pid = q["pair_id"]
        prompt = build_prompt(tokenizer, q["query"], cfg)
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        prompt_ids = enc["input_ids"][0]
        n = prompt_ids.shape[0]
        token_at_t = tokenizer.decode([prompt_ids[-1].item()])
        if token_at_t != "\n":
            raise RuntimeError(f"[{pid}] expected last prompt token '\\n', got {token_at_t!r}")

        # Generation; the recorder keeps only the prefill forward pass -> h.
        with rec.capture():
            out = model.generate(**enc, generation_config=gen_cfg)
        prefill = rec.layers()
        assert prefill[0].shape[0] == n, "prefill did not cover the full prompt"
        h = torch.stack([x[-1] for x in prefill]).float().cpu()  # [L+1, d]

        full_ids = out[0]
        gen_ids = full_ids[n:]
        m = gen_ids.shape[0]

        # Teacher-forced pass over prompt + generation -> Z(Q), plus a check on h.
        with rec.capture(), torch.no_grad():
            model(input_ids=full_ids.unsqueeze(0), use_cache=False)
        tf = rec.layers()
        h_tf = torch.stack([x[n - 1] for x in tf]).float().cpu()
        cos = torch.nn.functional.cosine_similarity(h, h_tf, dim=-1)
        h_check = {
            "max_abs_diff": (h - h_tf).abs().max().item(),
            "min_cosine": cos.min().item(),
            "min_cosine_layer": int(cos.argmin().item()),
        }

        for i in range(L + 1):
            torch.save(h[i].clone(), os.path.join(layer_dir(out_root, "h", i), f"{pid}.pt"))
            torch.save(tf[i][n:].to(traj_dtype).cpu().clone(),
                       os.path.join(layer_dir(out_root, "Z", i), f"{pid}.pt"))
            h_all[i].append(h[i].clone())

        reasoning = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        finished = m > 0 and gen_ids[-1].item() in eos_ids
        pred = parse_answer(reasoning)
        match = auto_match(pred, q["reference_answer"])

        entries.append({
            **q,
            "prompt": prompt,
            "prompt_ids": prompt_ids.tolist(),
            "num_prompt_tokens": n,
            "position_t": n - 1,
            "token_at_t": token_at_t,
            "generated_ids": gen_ids.tolist(),
            "num_generated_tokens": m,
            "finished": finished,
            "reasoning_text": reasoning,
            "predicted_answer": pred,
            "auto_match": match,
            "h_check": h_check,
        })
        flag = "" if h_check["min_cosine"] >= H_COSINE_WARN else "  [WARN h mismatch]"
        print(f"[{pid}] prompt={n} gen={m} finished={finished} "
              f"pred={pred!r} ref={q['reference_answer']!r} match={match} "
              f"h_cos_min={h_check['min_cosine']:.5f}{flag}")

    rec.remove()

    pair_ids = [e["pair_id"] for e in entries]
    for i in range(L + 1):
        torch.save({"pair_ids": pair_ids, "h": torch.stack(h_all[i])},
                   os.path.join(layer_dir(out_root, "h", i), "all.pt"))

    with open(os.path.join(out_root, "baseline.json"), "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    n_q = len(entries)
    summary = {
        "num_queries": n_q,
        "auto_accuracy": sum(e["auto_match"] for e in entries) / n_q,
        "num_unparsed": sum(e["predicted_answer"] is None for e in entries),
        "num_truncated": sum(not e["finished"] for e in entries),
        "mismatches": [e["pair_id"] for e in entries if not e["auto_match"]],
        "worst_h_min_cosine": min(e["h_check"]["min_cosine"] for e in entries),
        "worst_h_max_abs_diff": max(e["h_check"]["max_abs_diff"] for e in entries),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    metadata = {
        "config": cfg,
        "smoke": args.smoke,
        "max_new_tokens": max_new_tokens,
        "num_layers": L,
        "hidden_size": d,
        "layer_convention": "layer_00 = embedding output; layer_k = output of decoder block k-1 "
                            "(raw residual stream, no final norm)",
        "h_convention": "residual state at the last prompt token (position_t), taken from the "
                        "prefill of the generation call",
        "Z_convention": "Z/layer_XX/<id>.pt row k = residual state at generated token k, from a "
                        "teacher-forced pass over prompt + generation",
        "summary": summary,
    }
    with open(os.path.join(out_root, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\n=== Stage 1 summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    if summary["worst_h_min_cosine"] < H_COSINE_WARN:
        print("WARNING: prefill h and teacher-forced h disagree; inspect h_check in baseline.json")


if __name__ == "__main__":
    main()
