"""
Stage 2 -- soft rollout T_K(h) and reasoning metric G_R (Algorithm 1, steps 3-4).

Stage 2 validates the rollout machinery that stages 3-5 build on. It does not yet solve
for directions (that is stage 3). For each query it reports:

  A. h check     : h recomputed through the rollout path vs the saved stage 1 h.
  B. Fidelity    : roll out K steps unperturbed and compare with the model's real greedy
                   reasoning from stage 1 -- top token vs real token at each step, and
                   cos(s_k, Z[k-1]) at the measurement layer. "hard" mode (feeding argmax
                   tokens) must reproduce greedy decoding; the soft temperatures show how
                   far each T_s stays faithful. Chooses T_s and confirms K.
  C. Derivatives (first grad_check.num_queries queries), at the full per-query K:
       - adjoint test      <u, J v>  vs  <J^T u, v>
       - one-pass vs two-pass G_R v, symmetry <w, G v> vs <G w, v>, v^T G v = ||J v||^2
       - time and peak GPU memory of one G_R v product
  D. bf16 precision (if grad_check.precision_check): reload the model in fp32 and recompute
     J v for the same directions. Finite differences cannot test this in bf16 (rounding noise
     swamps the perturbation), so the fp32 product is the reference.

Outputs (under <output_root>/stage_2/<split>/): fidelity.json, grad_checks.json, summary.json

Run from the project root:
    python stage_2/run_stage2.py            # full
    python stage_2/run_stage2.py --smoke    # 1 query, K_max=16
"""

import argparse
import copy
import gc
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import layer_dir, load_config, load_model_and_tokenizer, resolve  # noqa: E402
from rollout import SoftRollout  # noqa: E402

cos1d = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=0).item()  # noqa: E731


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--smoke", action="store_true", help="1 query, K_max=16")
    ap.add_argument("--split", choices=["original", "counterfactual"], default=None)
    return ap.parse_args()


def rel(a, b):
    """|a - b| / max(|a|, |b|) for scalars."""
    return abs(a - b) / max(abs(a), abs(b), 1e-30)


def gb(nbytes):
    return round(nbytes / 2**30, 2)


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def fidelity_run(ro, prompt_ids, K, real_ids, Z, windows):
    with torch.no_grad():
        S, diag = ro.run(prompt_ids, K, collect=True)
    agree = [a == r for a, r in zip(diag["argmax"], real_ids[:K])]
    cos = torch.nn.functional.cosine_similarity(S.float(), Z[:K], dim=-1).tolist()
    first_disagree = next((i for i, ok in enumerate(agree) if not ok), K)
    win = {}
    for w in windows:
        if w <= K:
            win[str(w)] = {"agree": sum(agree[:w]) / w, "cos": sum(cos[:w]) / w}
    return {
        "K": K,
        "first_disagreement": first_disagree,
        "agree_fraction": sum(agree) / K,
        "mean_cos": sum(cos) / K,
        "mean_entropy": sum(diag["entropy"]) / K,
        "windows": win,
        "per_step": {"agree": agree, "cos": [round(c, 5) for c in cos],
                     "entropy": [round(e, 4) for e in diag["entropy"]]},
    }


def grad_checks(ro, prompt_ids, h0, K, seed):
    """Derivative checks at h0. Returns (json-able results, tensors for the precision check)."""
    device = h0.device
    d = h0.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)

    def unit(n):
        x = torch.randn(n, generator=g)
        return (x / x.norm()).to(device)

    v, w, u = unit(d), unit(d), unit(K * d)
    h0 = h0.float()
    res = {"K": K, "h_norm": h0.norm().item()}

    def matvec(vec):
        """One-pass G_R v; falls back to two passes if forward+reverse AD cannot combine."""
        if "one_pass_error" not in res:
            try:
                return ro.gr_matvec(prompt_ids, h0, K, vec)
            except Exception as ex:  # recorded so stage 3 knows which product to use
                res["one_pass_error"] = f"{type(ex).__name__}: {ex}"
                print(f"  one-pass G_R v failed, using two-pass: {res['one_pass_error']}")
        return ro.gr_matvec_two_pass(prompt_ids, h0, K, vec)

    # Injecting h0 itself must leave the rollout unchanged.
    with torch.no_grad():
        T_none = ro.trajectory(prompt_ids, K)
        T_h0 = ro.trajectory(prompt_ids, K, h=h0)
    res["inject_identity_rel_diff"] = ((T_none - T_h0).norm() / T_none.norm()).item()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    Gv = matvec(v)
    torch.cuda.synchronize()
    res["matvec_seconds"] = round(time.time() - t0, 2)
    res["matvec_peak_gb"] = gb(torch.cuda.max_memory_allocated())
    res["matvec_method"] = "two_pass" if "one_pass_error" in res else "one_pass"

    Gv2 = ro.gr_matvec_two_pass(prompt_ids, h0, K, v)
    res["one_vs_two_pass_rel_diff"] = ((Gv - Gv2).norm() / Gv2.norm()).item()

    T0, Jv = ro.jvp(prompt_ids, h0, K, v)
    res["primal_vs_plain_rel_diff"] = ((T0 - T_none).norm() / T_none.norm()).item()
    JTu = ro.vjp(prompt_ids, h0, K, u)
    a, b = torch.dot(u, Jv).item(), torch.dot(JTu, v).item()
    res["adjoint"] = {"u_Jv": a, "JTu_v": b, "rel_diff": rel(a, b)}

    vGv, Jv_sq = torch.dot(v, Gv).item(), Jv.norm().item() ** 2
    res["vGv_vs_Jv_sq"] = {"vGv": vGv, "Jv_sq": Jv_sq, "rel_diff": rel(vGv, Jv_sq)}
    res["rayleigh_random_v"] = vGv  # scale of G_R along a random unit direction

    Gw = matvec(w)
    a, b = torch.dot(w, Gv).item(), torch.dot(Gw, v).item()
    res["symmetry"] = {"w_Gv": a, "Gw_v": b, "rel_diff": rel(a, b)}

    stash = {"prompt_ids": prompt_ids.cpu(), "K": K, "v": v.cpu(), "h0": h0.cpu(),
             "T": T0.cpu(), "Jv": Jv.cpu()}
    return res, stash


def precision_check(ro32, stash):
    """Compare bf16 quantities with the same computation in fp32."""
    device = ro32.E.device
    prompt_ids = stash["prompt_ids"].to(device)
    h32 = ro32.base_h(prompt_ids).float()
    T32, Jv32 = ro32.jvp(prompt_ids, h32, stash["K"], stash["v"].to(device))
    T32, Jv32 = T32.cpu(), Jv32.cpu()
    return {
        "h_cos": cos1d(stash["h0"], h32.cpu()),
        "T_cos": cos1d(stash["T"], T32),
        "Jv_cos": cos1d(stash["Jv"], Jv32),
        "Jv_rel_err": ((stash["Jv"] - Jv32).norm() / Jv32.norm()).item(),
        "Jv_norm_ratio_bf16_over_fp32": (stash["Jv"].norm() / Jv32.norm()).item(),
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.split:
        cfg["data"]["split"] = args.split
    split = cfg["data"]["split"]
    s2 = cfg["stage_2"]
    gcfg = s2["grad_check"]

    num_queries = cfg["data"]["num_queries"]
    K_max = s2["horizon_K_max"]
    if args.smoke:
        num_queries = s2["smoke"]["num_queries"]
        K_max = s2["smoke"]["horizon_K_max"]
    out_root = resolve(cfg["paths"]["output_root"])
    s1_dir = os.path.join(out_root, "stage_1", split)
    out_dir = os.path.join(out_root, "stage_2_smoke" if args.smoke else "stage_2", split)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(s1_dir, "baseline.json"), encoding="utf-8") as f:
        entries = json.load(f)[:num_queries]

    model, _ = load_model_and_tokenizer(cfg, attn_implementation="eager")
    device = model.device
    ro = SoftRollout(model, s2["layer"], s2["measurement_layer"])
    temps = s2["fidelity"]["temperatures"]
    windows = s2["fidelity"]["windows"]
    print(f"layer={s2['layer']} measurement_layer={s2['measurement_layer']} K_max={K_max} "
          f"per_query={s2['horizon_per_query']} T_s={s2['soft_temperature']} split={split} "
          f"dtype={cfg['model']['dtype']}")

    fidelity, checks, stashes = [], [], []
    peak_overall = 0  # grad checks reset CUDA peak stats, so track the max here
    for qi, e in enumerate(entries):
        pid = e["pair_id"]
        prompt_ids = torch.tensor([e["prompt_ids"]], device=device)
        real_ids = e["generated_ids"]
        K = min(K_max, len(real_ids)) if s2["horizon_per_query"] else K_max
        Z = torch.load(os.path.join(layer_dir(s1_dir, "Z", s2["measurement_layer"]), f"{pid}.pt"))
        Z = Z.to(device).float()
        saved_h = torch.load(os.path.join(layer_dir(s1_dir, "h", s2["layer"]), f"{pid}.pt"))

        ro.mode, ro.temperature = "soft", s2["soft_temperature"]
        h0 = ro.base_h(prompt_ids)
        h_cos = cos1d(h0.float().cpu(), saved_h)

        rec = {"pair_id": pid, "K": K, "h_cos_vs_stage1": h_cos, "modes": {}}
        for mode, temp in [("hard", None)] + [("soft", t) for t in temps]:
            ro.mode, ro.temperature = mode, temp if temp is not None else 1.0
            name = "hard" if mode == "hard" else f"soft_T{temp}"
            rec["modes"][name] = fidelity_run(ro, prompt_ids, K, real_ids, Z, windows)
        fidelity.append(rec)
        line = " ".join(f"{n}:agree={m['agree_fraction']:.2f},1st_miss={m['first_disagreement']}"
                        for n, m in rec["modes"].items())
        print(f"[{pid}] K={K} h_cos={h_cos:.5f} {line}")

        peak_overall = max(peak_overall, torch.cuda.max_memory_allocated())
        if qi < gcfg["num_queries"]:
            ro.mode, ro.temperature = "soft", s2["soft_temperature"]
            res, stash = grad_checks(ro, prompt_ids, h0, K, gcfg["seed"] + qi)
            peak_overall = max(peak_overall, torch.cuda.max_memory_allocated())
            res["pair_id"] = pid
            checks.append(res)
            stashes.append(stash)
            print(f"[{pid}] grad: {res['matvec_method']} matvec {res['matvec_seconds']}s "
                  f"peak {res['matvec_peak_gb']}GB | adjoint {res['adjoint']['rel_diff']:.2e} | "
                  f"symmetry {res['symmetry']['rel_diff']:.2e} | "
                  f"1v2pass {res['one_vs_two_pass_rel_diff']:.2e}")
        torch.cuda.empty_cache()
    ro.remove()

    precision = []
    if gcfg.get("precision_check") and stashes and cfg["model"]["dtype"] != "float32":
        print("\nPrecision check: reloading the model in float32...")
        del ro, model
        gc.collect()
        torch.cuda.empty_cache()
        cfg32 = copy.deepcopy(cfg)
        cfg32["model"]["dtype"] = "float32"
        model32, _ = load_model_and_tokenizer(cfg32, attn_implementation="eager")
        ro32 = SoftRollout(model32, s2["layer"], s2["measurement_layer"],
                           temperature=s2["soft_temperature"])
        torch.cuda.reset_peak_memory_stats()
        for c, stash in zip(checks, stashes):
            p = precision_check(ro32, stash)
            p["pair_id"] = c["pair_id"]
            c["precision_vs_fp32"] = p
            precision.append(p)
            print(f"[{p['pair_id']}] bf16 vs fp32: h_cos={p['h_cos']:.5f} T_cos={p['T_cos']:.5f} "
                  f"Jv_cos={p['Jv_cos']:.4f} Jv_rel_err={p['Jv_rel_err']:.3f}")
        peak_overall = max(peak_overall, torch.cuda.max_memory_allocated())
        ro32.remove()

    fid_summary = {}
    for n in fidelity[0]["modes"]:
        ms = [r["modes"][n] for r in fidelity]
        fid_summary[n] = {
            "mean_first_disagreement": mean([m["first_disagreement"] for m in ms]),
            "mean_agree_fraction": mean([m["agree_fraction"] for m in ms]),
            "mean_cos": mean([m["mean_cos"] for m in ms]),
            "mean_entropy": mean([m["mean_entropy"] for m in ms]),
            "windows": {
                str(w): {
                    "n_queries": len([m for m in ms if str(w) in m["windows"]]),
                    "agree": mean([m["windows"][str(w)]["agree"] for m in ms if str(w) in m["windows"]]),
                    "cos": mean([m["windows"][str(w)]["cos"] for m in ms if str(w) in m["windows"]]),
                } for w in windows
            },
        }
    summary = {
        "config": s2,
        "split": split,
        "smoke": args.smoke,
        "K_max": K_max,
        "num_queries": len(fidelity),
        "worst_h_cos_vs_stage1": min(r["h_cos_vs_stage1"] for r in fidelity),
        "fidelity": fid_summary,
        "grad_checks": [
            {k: c[k] for k in ("pair_id", "K", "matvec_method", "matvec_seconds", "matvec_peak_gb",
                               "one_vs_two_pass_rel_diff", "inject_identity_rel_diff")}
            | {"adjoint_rel_diff": c["adjoint"]["rel_diff"],
               "symmetry_rel_diff": c["symmetry"]["rel_diff"]}
            for c in checks
        ],
        "precision_vs_fp32": precision,
        "peak_gpu_gb_overall": gb(max(peak_overall, torch.cuda.max_memory_allocated())),
    }
    for name, obj in [("fidelity.json", fidelity), ("grad_checks.json", checks),
                      ("summary.json", summary)]:
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)

    print("\n=== Stage 2 summary ===")
    print(f"worst h cos vs stage 1: {summary['worst_h_cos_vs_stage1']:.5f}")
    print(f"{'mode':<12}{'1st_miss':>9}{'agree':>7}{'cos':>7}{'entropy':>8}   agree@window")
    for n, s in fid_summary.items():
        wins = " ".join(f"{w}:{v['agree']:.2f}" for w, v in s["windows"].items() if v["agree"] is not None)
        print(f"{n:<12}{s['mean_first_disagreement']:>9.1f}{s['mean_agree_fraction']:>7.2f}"
              f"{s['mean_cos']:>7.3f}{s['mean_entropy']:>8.3f}   {wins}")
    for c in summary["grad_checks"]:
        print(f"grad {c['pair_id']}: K={c['K']} {c['matvec_method']} matvec={c['matvec_seconds']}s "
              f"peak={c['matvec_peak_gb']}GB adjoint={c['adjoint_rel_diff']:.2e} "
              f"symmetry={c['symmetry_rel_diff']:.2e} 1v2={c['one_vs_two_pass_rel_diff']:.2e} "
              f"inject_id={c['inject_identity_rel_diff']:.2e}")
    for p in precision:
        print(f"precision {p['pair_id']}: Jv_cos={p['Jv_cos']:.4f} Jv_rel_err={p['Jv_rel_err']:.3f} "
              f"norm_ratio={p['Jv_norm_ratio_bf16_over_fp32']:.3f}")
    print(f"peak GPU overall: {summary['peak_gpu_gb_overall']} GB")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
