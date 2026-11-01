"""E2 - Probing (representational level).

For each of the 96 E1 prompts (chat-templated), run a forward pass and
record the residual stream at the final prompt token for every layer
(embedding + 24 layers = 25 states, hidden size 896).

Per layer:
  (a) multinomial logistic-regression probe (numpy softmax regression,
      Adam, 300 epochs, L2 1e-3, features standardised with train-fold
      statistics) classifying the operator (3-way; chance = 1/3),
      5-fold cross-validation (seeded, seed 42). scikit-learn was not
      available in the reference environment, hence the numpy probe.
  (b) ridge regression (closed form, lambda 1.0, standardised) of the
      ground-truth multiplier, 5-fold CV, reporting mean R^2.

The hidden-state cache is written to disk after EVERY prompt
(results/e2_cache_partial.npz, float16), so the collection phase is
resumable across process/host restarts: prompts already in the cache
are skipped on rerun.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from common import (OPERATORS, PRICES, RATES, chat_prompt, dump_json,  # noqa: E402
                    ground_truth_multiplier, load_model)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "results", "e2_results.json")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "results", "e2_hidden_cache.npz")
PARTIAL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "results", "e2_cache_partial.npz")
NLAYERS, HID = 25, 896


def collect(tok, model):
    items = [(op, r, p) for op in OPERATORS for r in RATES for p in PRICES]
    y_op = np.array([OPERATORS.index(op) for op, _, _ in items])
    y_mult = np.array([ground_truth_multiplier(op, r) for op, r, _ in items],
                      dtype=np.float64)
    X = np.zeros((len(items), NLAYERS, HID), dtype=np.float16)
    done = np.zeros(len(items), dtype=bool)
    if os.path.exists(PARTIAL):
        z = np.load(PARTIAL)
        X, done = z["X"], z["done"]
        # Invalidate poisoned rows: on the shared host used for these
        # experiments, some forward passes intermittently returned NaN
        # hidden states in layers >= 1 (39 of 96 rows in the first
        # collection attempts). A cached row only counts as done if it
        # is fully finite.
        bad = 0
        for i in range(len(items)):
            if done[i] and not np.isfinite(X[i]).all():
                done[i] = False
                bad += 1
        print(f"resuming cache: {int(done.sum())}/{len(items)} present "
              f"({bad} non-finite rows invalidated)")
    import time as _time
    for i, (op, r, p) in enumerate(items):
        if done[i]:
            continue
        prompt = chat_prompt(tok, op, r, p)
        ids = tok(prompt, return_tensors="pt")
        hs = None
        for attempt in range(5):
            with torch.no_grad():
                out = model(**ids, output_hidden_states=True)
            cand = np.stack([h[0, -1].float().numpy()
                             for h in out.hidden_states])  # (25, 896)
            if np.isfinite(cand).all():
                hs = cand
                break
            print(f"E2 collect {i + 1}: non-finite hidden states "
                  f"(attempt {attempt + 1}), retrying", flush=True)
            _time.sleep(2)
        if hs is None:
            print(f"E2 collect {i + 1}: STILL non-finite after retries; "
                  f"row left undone", flush=True)
            continue
        X[i] = hs.astype(np.float16)
        done[i] = True
        np.savez(PARTIAL, X=X, done=done)
        print(f"E2 collect {i + 1}/{len(items)}", flush=True)
    finite_rows = np.array([bool(np.isfinite(X[i]).all()) and done[i]
                            for i in range(len(items))])
    if not finite_rows.all():
        dropped = [items[i] for i in range(len(items))
                   if not finite_rows[i]]
        print(f"E2: {len(dropped)} prompts dropped (non-finite): "
              f"{dropped}")
        np.savez(CACHE, X=X[finite_rows], y_op=y_op[finite_rows],
                 y_mult=y_mult[finite_rows])
        return (X[finite_rows].astype(np.float64), y_op[finite_rows],
                y_mult[finite_rows], dropped)
    np.savez(CACHE, X=X, y_op=y_op, y_mult=y_mult)
    return X.astype(np.float64), y_op, y_mult, []


def softmax_probs(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def train_softmax(Xtr, ytr, nclass=3, epochs=300, lr=0.05, l2=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.01, size=(Xtr.shape[1], nclass))
    b = np.zeros(nclass)
    mW = np.zeros_like(W); vW = np.zeros_like(W)
    mb = np.zeros_like(b); vb = np.zeros_like(b)
    Y = np.eye(nclass)[ytr]
    n = len(ytr)
    for t in range(1, epochs + 1):
        P = softmax_probs(Xtr @ W + b)
        G = (P - Y) / n
        gW = Xtr.T @ G + l2 * W
        gb = G.sum(axis=0)
        for param, grad, m, v in ((W, gW, mW, vW), (b, gb, mb, vb)):
            m[:] = 0.9 * m + 0.1 * grad
            v[:] = 0.999 * v + 0.001 * grad * grad
            mh = m / (1 - 0.9 ** t)
            vh = v / (1 - 0.999 ** t)
            param -= lr * mh / (np.sqrt(vh) + 1e-8)
    return W, b


def cv_folds(n, k=5, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return np.array_split(idx, k)


def probe_layer(Xl, y_op, y_mult, folds):
    accs, r2s = [], []
    for test_idx in folds:
        mask = np.ones(len(y_op), dtype=bool)
        mask[test_idx] = False
        tr = np.where(mask)[0]
        mu, sd = Xl[tr].mean(0), Xl[tr].std(0) + 1e-8
        Xtr = (Xl[tr] - mu) / sd
        Xte = (Xl[test_idx] - mu) / sd
        W, b = train_softmax(Xtr, y_op[tr])
        pred = (Xte @ W + b).argmax(1)
        accs.append(float((pred == y_op[test_idx]).mean()))
        A = np.concatenate([Xtr, np.ones((len(tr), 1))], axis=1)
        w = np.linalg.solve(
            A.T @ A + 1.0 * np.eye(A.shape[1]),
            A.T @ y_mult[tr])
        B = np.concatenate([Xte, np.ones((len(test_idx), 1))], axis=1)
        pm = B @ w
        yt = y_mult[test_idx]
        ss_res = float(((yt - pm) ** 2).sum())
        ss_tot = float(((yt - yt.mean()) ** 2).sum())
        r2s.append(1.0 - ss_res / ss_tot)
    return float(np.mean(accs)), float(np.mean(r2s))


def main():
    tok, model = load_model()
    X, y_op, y_mult, dropped = collect(tok, model)
    folds = cv_folds(len(y_op))
    acc, r2 = [], []
    for L in range(NLAYERS):
        a, r = probe_layer(X[:, L, :], y_op, y_mult, folds)
        acc.append(a)
        r2.append(r)
        print(f"layer {L}: acc {a:.4f} R2 {r:.4f}", flush=True)
    best = int(np.argmax(acc))
    dump_json({
        "experiment": "E2", "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "n_prompts": int(len(y_op)), "position": "final prompt token",
        "dropped_prompts_nonfinite": [list(map(str, d)) for d in dropped],
        "layer_operator_accuracy": acc,
        "layer_multiplier_R2": r2,
        "best_layer_accuracy": best,
        "best_layer_multiplier_R2": int(np.argmax(r2)),
        "chance_accuracy": 1.0 / 3.0,
        "probe": "numpy softmax regression, Adam 300 epochs, lr 0.05, "
                 "L2 1e-3; ridge lambda 1.0; 5-fold CV seed 42",
    }, RESULTS)
    print("E2 best layer:", best, "acc:", acc[best])


if __name__ == "__main__":
    main()
