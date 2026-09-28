"""
Differentiable soft rollout T_K(h) and the reasoning metric G_R = J_R^T J_R (PDF Sec. 1.3).

Intervention: the residual stream at layer_l (output of decoder block l-1, same convention
as stage 1) at position t, the last prompt token, is replaced by h. Everything else about
the prompt is unchanged: positions 0..t-1 are computed normally, and layers < l at
position t keep their original values (so their KV entries do not depend on h).

Rollout (Eq. 8-12), starting from the logits produced at position t:
    p_k   = softmax(z_k / T_s)                         (Eq. 9)
    e_k   = E^T p_k   (probability-weighted embedding)  (Eq. 10)
    feed e_k as the next input embedding, read s_k = residual at the measurement layer
    T_K(h) = concat(s_1..s_K) / sqrt(K)                 (Eq. 12)

s_k is aligned with stage 1's Z(Q): s_k <-> Z[k-1] (the state at the k-th generated
token). mode="hard" feeds the argmax token's embedding instead, which reproduces greedy
decoding and is used as a sanity check.

Derivative products, all without forming J_R:
    jvp(v)       = J_R v              forward-mode AD
    vjp(u)       = J_R^T u            reverse-mode AD
    gr_matvec(v) = J_R^T (J_R v)      one rollout: forward AD for J_R v, then backward (Eq. 22)
    gr_matvec_two_pass(v)             same product via jvp + vjp (two rollouts), cross-check

The model must be loaded with attn_implementation="eager".
"""

import math

import torch
import torch.autograd.forward_ad as fwAD


def _hidden(out):
    return out[0] if isinstance(out, tuple) else out


def _replace_hidden(out, new):
    return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new


class SoftRollout:
    def __init__(self, model, layer, measurement_layer, temperature=1.0, mode="soft"):
        n_layers = len(model.model.layers)
        if not (1 <= layer <= n_layers and 1 <= measurement_layer <= n_layers):
            raise ValueError(f"layer and measurement_layer must be in 1..{n_layers}")
        if mode not in ("soft", "hard"):
            raise ValueError("mode must be 'soft' or 'hard'")
        self.model = model
        self.layer = layer
        self.measurement_layer = measurement_layer
        self.temperature = temperature
        self.mode = mode
        self.E = model.get_input_embeddings().weight  # [V, d], frozen
        self.dtype = self.E.dtype

        self._step0 = False     # True only during the forward pass over position t
        self._inject = None     # tensor substituted at layer_l, position t (None = keep)
        self.original_h = None  # unperturbed residual at layer_l, position t (detached)
        self._measuring = False
        self._meas = []

        blocks = model.model.layers
        self._handles = [
            blocks[layer - 1].register_forward_hook(self._inject_hook),
            blocks[measurement_layer - 1].register_forward_hook(self._measure_hook),
        ]

    # -- hooks ---------------------------------------------------------------------------
    def _inject_hook(self, _module, _inp, out):
        if not self._step0:
            return None
        hs = _hidden(out)
        assert hs.shape[1] == 1, "step-0 forward must cover position t only"
        self.original_h = hs[0, -1].detach()
        self._step0 = False
        if self._inject is None:
            return None
        new = self._inject.reshape(1, 1, -1).to(hs.dtype)
        self._inject = None
        return _replace_hidden(out, new)

    def _measure_hook(self, _module, _inp, out):
        if self._measuring:
            self._meas.append(_hidden(out)[0, -1])

    def remove(self):
        for h in self._handles:
            h.remove()

    # -- rollout -------------------------------------------------------------------------
    def _next_embedding(self, logits):
        logits = logits.float()
        if self.mode == "hard":
            idx = int(logits.argmax())
            return self.E[idx], idx, 0.0
        p = torch.softmax(logits / self.temperature, dim=-1)
        with torch.no_grad():
            entropy = float(-(p * torch.log(p.clamp_min(1e-30))).sum())
        return p.to(self.dtype) @ self.E, int(p.argmax()), entropy

    def run(self, prompt_ids, K, h=None, collect=False):
        """Roll out K soft steps. prompt_ids: [1, n] on the model device.

        Returns S [K, d] (attached to h's graph / dual tangent when h is given).
        With collect=True also returns per-step diagnostics:
            argmax[k-1]  = the rollout's most likely token at step k
            entropy[k-1] = entropy of p_k (nats); 0 in hard mode
        """
        model = self.model
        with torch.no_grad():
            cache = model(input_ids=prompt_ids[:, :-1], use_cache=True).past_key_values

        self._step0, self._inject = True, h
        out = model(input_ids=prompt_ids[:, -1:], past_key_values=cache, use_cache=True)
        if self._step0 or self._inject is not None:
            self._step0, self._inject = False, None
            raise RuntimeError("injection hook did not fire at position t")
        logits, cache = out.logits[0, -1], out.past_key_values

        argmax, entropy = [], []
        self._meas, self._measuring = [], True
        try:
            for _ in range(K):
                emb, top, ent = self._next_embedding(logits)
                argmax.append(top)
                entropy.append(ent)
                out = model(inputs_embeds=emb.reshape(1, 1, -1),
                            past_key_values=cache, use_cache=True)
                logits, cache = out.logits[0, -1], out.past_key_values
        finally:
            self._measuring = False
        d = self.E.shape[1]
        S = torch.stack(self._meas) if self._meas else self.E.new_zeros(0, d)
        self._meas = []
        if collect:
            return S, {"argmax": argmax, "entropy": entropy}
        return S

    def trajectory(self, prompt_ids, K, h=None):
        """T_K(h) as a flat fp32 vector of length K*d (Eq. 12)."""
        return self.run(prompt_ids, K, h=h).float().reshape(-1) / math.sqrt(K)

    # -- derivative products ---------------------------------------------------------------
    def base_h(self, prompt_ids):
        """Unperturbed h at (layer_l, t), computed by this model in its own dtype."""
        with torch.no_grad():
            self.run(prompt_ids, 0)
        return self.original_h.clone()

    def jvp(self, prompt_ids, h0, K, v):
        """(T_K(h0), J_R v), both fp32. No reverse graph is built: nothing requires grad."""
        with fwAD.dual_level():
            hd = fwAD.make_dual(h0.to(self.dtype), v.to(self.dtype))
            T = self.trajectory(prompt_ids, K, h=hd)
            primal, tangent = fwAD.unpack_dual(T)
            return primal.clone(), tangent.clone()

    def vjp(self, prompt_ids, h0, K, u):
        """J_R^T u, fp32 [d]."""
        h = h0.detach().to(self.dtype).requires_grad_(True)
        T = self.trajectory(prompt_ids, K, h=h)
        (g,) = torch.autograd.grad(T, h, grad_outputs=u.to(T.dtype))
        return g.float()

    def gr_matvec(self, prompt_ids, h0, K, v):
        """G_R v = J_R^T (J_R v) with a single rollout (Eq. 22). Returns fp32 [d]."""
        h = h0.detach().to(self.dtype).requires_grad_(True)
        with fwAD.dual_level():
            hd = fwAD.make_dual(h, v.to(self.dtype))
            T = self.trajectory(prompt_ids, K, h=hd)
            primal, tangent = fwAD.unpack_dual(T)
            Jv = tangent.detach().clone()
        # Backward must run outside the dual level, or autograd tries to forward-differentiate
        # the backward ops (e.g. silu_backward has no forward-AD formula).
        (g,) = torch.autograd.grad(primal, h, grad_outputs=Jv)
        return g.float()

    def gr_matvec_two_pass(self, prompt_ids, h0, K, v):
        """Same product with two rollouts (jvp then vjp); slower, used as a cross-check."""
        _, Jv = self.jvp(prompt_ids, h0, K, v)
        return self.vjp(prompt_ids, h0, K, Jv)
