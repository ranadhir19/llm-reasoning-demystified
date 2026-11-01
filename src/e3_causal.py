"""E3 - Causal interventions.

Measurement unit: a plain-text setup ending in "The equation is PRICE =",
followed by the target continuation " m * P." The score of a continuation
is its teacher-forced log-probability under the model.

(a) Operator ablation: the operator word is replaced by a neutral wording
    ("discount"/"increase" -> "change"; the "of" construction ->
    "relative to") and the drop in the correct multiplier's log-probability
    is measured. Subset: rates {10,20,30,50} x 3 operators, price 120.

(b) Activation patching: for discount<->increase pairs (same rate, price
    120), the residual stream at the operator-token position(s) is copied
    layer by layer from the source run into the target run. The effect is
    the shift, at the answer position, of
        logit(first token of source multiplier) - logit(first token of
        target multiplier)
    normalised by the intact source-minus-target gap. Both directions are
    run and averaged. Peak layer is reported.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from common import dump_json, ground_truth_multiplier, load_model  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "results", "e3_results.json")
RATES_SUB = [10, 20, 30, 50]
PRICE = 120


def stem(operator, rate, price, ablated=False):
    if operator == "discount":
        w = "change" if ablated else "discount"
        return (f"A shop sells a product for ${price} after applying a "
                f"{rate}% {w}.")
    if operator == "increase":
        w = "change" if ablated else "increase"
        return (f"A shop sells a product for ${price} after applying a "
                f"{rate}% {w}.")
    if operator == "of":
        if ablated:
            return (f"A shop sells a product for ${price}, which is {rate}% "
                    f"relative to its original price.")
        return (f"A shop sells a product for ${price}, which is {rate}% "
                f"of its original price.")
    raise ValueError(operator)


def setup_prompt(operator, rate, price, ablated=False):
    return (stem(operator, rate, price, ablated)
            + f" Let P be the original price. The equation is {price} =")


def continuation(operator, rate):
    m = ground_truth_multiplier(operator, rate)
    return f" {m:g} * P."


def seq_logprob(tok, model, prompt, cont):
    pids = tok(prompt, return_tensors="pt")["input_ids"]
    cids = tok(cont, add_special_tokens=False, return_tensors="pt")["input_ids"]
    ids = torch.cat([pids, cids], dim=1)
    with torch.no_grad():
        logits = model(ids).logits[0]
    lp = torch.log_softmax(logits.float(), dim=-1)
    start = pids.shape[1]
    tot = 0.0
    for i in range(cids.shape[1]):
        tot += lp[start + i - 1, cids[0, i]].item()
    # On the shared host used here, forwards intermittently returned
    # non-finite values; fail loudly so the supervisor retries the
    # whole experiment in a fresh process instead of recording garbage.
    if not np.isfinite(tot):
        raise RuntimeError("non-finite logprob in E3")
    return tot


def operator_positions(tok, text, word):
    enc = tok(text, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    start = text.index(word)
    end = start + len(word)
    return [i for i, (a, b) in enumerate(offs) if a < end and b > start]


def last_logits(tok, model, prompt):
    ids = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        return model(**ids).logits[0, -1]


def first_mult_token(tok, operator, rate):
    """First DIGIT token of the multiplier string.

    Fix (2026-11-01): the original version took ids[0] of " 0.9", which is
    the standalone space token (id 220) for every multiplier, so the
    source/target score difference was identically zero and the whole
    patching curve collapsed to 0. Scoring now uses the first digit token
    ('0' = 15 vs '1' = 16), and prompts are scored with a trailing space
    appended so the scored position is the one that predicts that digit.
    """
    s = f"{ground_truth_multiplier(operator, rate):g}"
    digit = next(ch for ch in s if ch.isdigit())
    ids = tok(digit, add_special_tokens=False)["input_ids"]
    return ids[0], digit


def patching_for_pair(tok, model, src_op, tgt_op, rate):
    src_prompt = setup_prompt(src_op, rate, PRICE)
    tgt_prompt = setup_prompt(tgt_op, rate, PRICE)
    src_pos = operator_positions(tok, src_prompt, src_op)
    tgt_pos = operator_positions(tok, tgt_prompt, tgt_op)
    src_tok, _ = first_mult_token(tok, src_op, rate)
    tgt_tok, _ = first_mult_token(tok, tgt_op, rate)

    def score(logits):
        return (logits[src_tok] - logits[tgt_tok]).item()

    ids_s = tok(src_prompt + " ", return_tensors="pt")
    with torch.no_grad():
        src_out = model(**ids_s, output_hidden_states=True)
    src_h = src_out.hidden_states  # index L+1 = residual after layer L
    base_src = score(src_out.logits[0, -1])
    base_tgt = score(last_logits(tok, model, tgt_prompt + " "))
    gap = base_src - base_tgt
    n_layers = len(model.model.layers)
    effects = []
    for L in range(n_layers):
        src_vals = src_h[L + 1][0]  # (seq, d)
        tgt_ids = tok(tgt_prompt + " ", return_tensors="pt")

        def hook(module, inp, out, L=L, src_vals=src_vals):
            h = out[0].clone()
            for j, tp in enumerate(tgt_pos):
                sp = src_pos[min(j, len(src_pos) - 1)]
                h[0, tp] = src_vals[sp]
            return (h,) + out[1:]

        handle = model.model.layers[L].register_forward_hook(hook)
        try:
            with torch.no_grad():
                lg = model(**tgt_ids).logits[0, -1]
        finally:
            handle.remove()
        patched = score(lg)
        effects.append((patched - base_tgt) / gap if gap != 0 else 0.0)
    return effects, {"src_positions": src_pos, "tgt_positions": tgt_pos,
                     "gap": gap}


def main():
    tok, model = load_model()
    # ---- (a) ablation ----
    abl_rows = []
    for op in ["discount", "increase", "of"]:
        for rate in RATES_SUB:
            cont = continuation(op, rate)
            lp_ok = seq_logprob(tok, model, setup_prompt(op, rate, PRICE), cont)
            lp_ab = seq_logprob(tok, model,
                                setup_prompt(op, rate, PRICE, ablated=True),
                                cont)
            abl_rows.append({
                "operator": op, "rate": rate,
                "logprob_intact": lp_ok, "logprob_ablated": lp_ab,
                "logprob_drop": lp_ok - lp_ab,
                "prob_intact": float(np.exp(lp_ok)),
                "prob_ablated": float(np.exp(lp_ab)),
            })
            print(f"abl {op} {rate}: {lp_ok:.3f} -> {lp_ab:.3f}", flush=True)
    abl_by_op = {}
    for op in ["discount", "increase", "of"]:
        ds = [r["logprob_drop"] for r in abl_rows if r["operator"] == op]
        abl_by_op[op] = float(np.mean(ds))
    abl_mean = float(np.mean([r["logprob_drop"] for r in abl_rows]))

    # ---- (b) patching ----
    per_rate = {}
    meta = {}
    for rate in RATES_SUB:
        e1, m1 = patching_for_pair(tok, model, "discount", "increase", rate)
        e2, m2 = patching_for_pair(tok, model, "increase", "discount", rate)
        per_rate[str(rate)] = {
            "discount_into_increase": e1,
            "increase_into_discount": e2,
            "mean": ((np.array(e1) + np.array(e2)) / 2).tolist(),
        }
        meta[str(rate)] = {"d2i": m1, "i2d": m2}
        print(f"patch rate {rate} done", flush=True)
    mean_curve = np.mean([per_rate[str(r)]["mean"] for r in RATES_SUB], axis=0)
    if not np.isfinite(mean_curve).all():
        raise RuntimeError("non-finite patching curve in E3")
    peak_layer = int(np.argmax(mean_curve))
    dump_json({
        "experiment": "E3", "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "ablation": {"rows": abl_rows, "mean_logprob_drop_by_operator":
                     abl_by_op, "mean_logprob_drop": abl_mean,
                     "subset": "rates {10,20,30,50} x 3 operators, price 120"},
        "patching": {"per_rate": per_rate, "mean_curve": mean_curve.tolist(),
                     "peak_layer": peak_layer,
                     "peak_effect": float(mean_curve[peak_layer]),
                     "meta": meta,
                     "note": "Effect is normalised: 0 = target run unchanged, "
                             "1 = full flip to the source operator's "
                             "multiplier token."},
    }, RESULTS)
    print("ABL mean drop", abl_mean, abl_by_op)
    print("PATCH peak layer", peak_layer, float(mean_curve[peak_layer]))


if __name__ == "__main__":
    main()
