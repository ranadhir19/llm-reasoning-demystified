"""F3-F6 from the saved results JSONs (system python / matplotlib only)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def f3():
    d = load("e1_results.json")
    ops = ["discount", "increase", "of"]
    rates = [5, 10, 15, 20, 25, 30, 40, 50]
    M = np.array([[d["summary"][op]["by_rate"][str(r)] for r in rates]
                  for op in ops])
    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(rates)), [f"{r}%" for r in rates])
    ax.set_yticks(range(len(ops)), ops)
    ax.set_xlabel("Rate d")
    ax.set_ylabel("Operator")
    for i in range(len(ops)):
        for j in range(len(rates)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] < 0.6 else "black", fontsize=8.5)
    ax.set_title("Figure F3. E1 multiplier accuracy by operator and rate")
    fig.colorbar(im, ax=ax, shrink=0.9, label="accuracy")
    fig.savefig(os.path.join(FIG, "F3_e1_heatmap.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)


def f4():
    d = load("e2_results.json")
    layers = range(len(d["layer_operator_accuracy"]))
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.plot(layers, d["layer_operator_accuracy"], "o-", ms=4,
            label="Operator classification accuracy (chance 0.33)")
    ax.plot(layers, d["layer_multiplier_R2"], "s-", ms=4,
            label="Multiplier value regression R2")
    ax.axhline(1 / 3, color="grey", ls="--", lw=1)
    ax.set_xlabel("Layer (0 = embedding)")
    ax.set_ylabel("Cross-validated score")
    ax.set_title("Figure F4. E2 linear probes on the final prompt token")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(FIG, "F4_e2_probes.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)


def f5():
    d = load("e3_results.json")
    curve = d["patching"]["mean_curve"]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.plot(range(len(curve)), curve, "o-", ms=4)
    ax.axhline(0, color="grey", lw=1)
    ax.axhline(1, color="grey", ls="--", lw=1)
    ax.set_xlabel("Patched layer (residual stream at the operator position)")
    ax.set_ylabel("Normalised effect on multiplier logit gap")
    ax.set_title("Figure F5. E3 activation patching: discount <-> increase")
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(FIG, "F5_e3_patching.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)


def f6():
    d = load("e4_results.json")
    lens = d["logit_lens"]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.plot(lens["layers"], lens["p_digit8"], "o-", ms=4,
            label='P("8") - first digit of 80 - at the digit position')
    ax.plot(lens["layers"], lens["p_digit2"], "s-", ms=4,
            label='P("2") - first digit of 20 - at the digit position')
    ax.set_xlabel("Layer (0 = embedding, final = output)")
    ax.set_ylabel("Probability under final norm + lm_head")
    ax.set_title("Figure F6. E4 logit lens on the canonical example")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(FIG, "F6_e4_logit_lens.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    f3(); f4(); f5(); f6()
    print("result figures done")
