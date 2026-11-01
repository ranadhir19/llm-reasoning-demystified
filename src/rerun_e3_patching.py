"""Re-run ONLY the E3 activation-patching block with the fixed scorer.

The first patching run scored the standalone space token (id 220) for both
multipliers, so every gap was exactly 0 and the curve was identically
zero. first_mult_token / scoring positions are fixed in e3_causal.py
(first digit token, prompt scored with a trailing space). The ablation
block of results/e3_results.json is valid and is preserved unchanged;
only the "patching" key is recomputed and replaced.
"""
import json
import os

import numpy as np

from common import dump_json, load_model
from e3_causal import RATES_SUB, patching_for_pair

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "..", "results", "e3_results.json")


def main():
    tok, model = load_model()
    per_rate, meta = {}, {}
    for rate in RATES_SUB:
        e1, m1 = patching_for_pair(tok, model, "discount", "increase", rate)
        e2, m2 = patching_for_pair(tok, model, "increase", "discount", rate)
        per_rate[str(rate)] = {
            "discount_into_increase": e1,
            "increase_into_discount": e2,
            "mean": ((np.array(e1) + np.array(e2)) / 2).tolist(),
        }
        meta[str(rate)] = {"d2i": m1, "i2d": m2}
        print(f"patch rate {rate} done; gaps d2i={m1['gap']:.3f} "
              f"i2d={m2['gap']:.3f}", flush=True)
    mean_curve = np.mean([per_rate[str(r)]["mean"] for r in RATES_SUB],
                         axis=0)
    if not np.isfinite(mean_curve).all():
        raise RuntimeError("non-finite patching curve in rerun")
    peak_layer = int(np.argmax(mean_curve))
    with open(PATH) as f:
        data = json.load(f)
    old_note = data["patching"].get("note", "")
    data["patching"] = {
        "per_rate": per_rate, "mean_curve": mean_curve.tolist(),
        "peak_layer": peak_layer,
        "peak_effect": float(mean_curve[peak_layer]),
        "meta": meta,
        "note": old_note,
        "scorer_fix": "First run scored the space token (id 220) for both "
                      "multipliers (gap identically 0, curve identically "
                      "zero). Re-run with first-digit-token scoring "
                      "(0 vs 1) at the position after a trailing space; "
                      "ablation block unchanged.",
    }
    dump_json(data, PATH)
    print("patching replaced; peak layer", peak_layer,
          "effect", round(float(mean_curve[peak_layer]), 4))
    print("mean curve:", [round(x, 3) for x in mean_curve.tolist()])


if __name__ == "__main__":
    main()
