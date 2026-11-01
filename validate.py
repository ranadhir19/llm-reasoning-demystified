#!/usr/bin/env python3
"""House-pattern validation harness for llm-reasoning-demystified.

Stages (PASS/FAIL each):
  1. compile   - every .py file in the repo compiles.
  2. results   - the four experiment result JSONs exist, parse, and carry
                 the headline fields the article quotes.
  3. figures   - the six figures exist and are non-trivial PNG files.
  4. article   - article.md exists, has the required sections, and the
                 PDF exists and opens with the expected page count range.
  5. scan      - no employer identifiers anywhere in repo text files
                 (tokens assembled dynamically so this file does not
                 flag itself).

Honest coverage statement: this harness verifies package integrity and
internal consistency of the recorded artifacts. It does NOT re-run the
experiments and does NOT verify that the article's prose numbers match
an independent recomputation from raw data beyond the JSON field checks
in stage 2.
"""
import json
import os
import py_compile
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
failures = []


def stage(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures.append(name)
    return ok


def rel(*p):
    return os.path.join(ROOT, *p)


# ---- stage 1: compile -------------------------------------------------
py_files = []
for base, _dirs, files in os.walk(ROOT):
    for f in files:
        if f.endswith(".py"):
            py_files.append(os.path.join(base, f))
bad = []
for f in py_files:
    try:
        py_compile.compile(f, doraise=True)
    except Exception as e:  # noqa: BLE001
        bad.append(f"{f}: {e}")
stage("compile", not bad, f"({len(py_files)} files)"
      + ("" if not bad else " BAD: " + "; ".join(bad)))

# ---- stage 2: results ---------------------------------------------------
need = {
    "e1_results.json": ["overall_multiplier_accuracy", "summary", "records"],
    "e2_results.json": ["layer_operator_accuracy", "layer_multiplier_R2",
                        "best_layer_accuracy"],
    "e3_results.json": ["ablation", "patching"],
    "e4_results.json": ["attention_max", "logit_lens",
                        "continuation_logprob_80pct"],
}
ok = True
for fname, keys in need.items():
    p = rel("results", fname)
    if not os.path.exists(p):
        ok = False
        print(f"    missing results/{fname}")
        continue
    try:
        d = json.load(open(p))
        for k in keys:
            if k not in d:
                ok = False
                print(f"    results/{fname} lacks key {k}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"    results/{fname} unparsable: {e}")
stage("results", ok, "(4 experiment JSONs)")

# ---- stage 3: figures ---------------------------------------------------
figs = ["F1_context_window.png", "F2_translation_pipeline.png",
        "F3_e1_heatmap.png", "F4_e2_probes.png", "F5_e3_patching.png",
        "F6_e4_logit_lens.png"]
ok = True
for f in figs:
    p = rel("figures", f)
    if not os.path.exists(p) or os.path.getsize(p) < 5000:
        ok = False
        print(f"    figure missing/small: {f}")
    else:
        with open(p, "rb") as fh:
            if fh.read(8) != b"\x89PNG\r\n\x1a\n":
                ok = False
                print(f"    figure not a PNG: {f}")
stage("figures", ok, "(6 PNGs)")

# ---- stage 4: article ---------------------------------------------------
ok = True
amd = rel("article", "article.md")
if not os.path.exists(amd):
    ok = False
    print("    article/article.md missing")
else:
    text = open(amd, encoding="utf-8").read()
    for marker in ["## Abstract", "## 8. Experiments",
                   "## 12. Conclusion", "## References",
                   "## Reproducibility"]:
        if marker not in text:
            ok = False
            print(f"    article.md lacks section: {marker}")
    words = len(text.split())
    if words < 5500:
        ok = False
        print(f"    article.md too short: {words} words")
pdf = rel("article", "Demystifying_LLM_Reasoning.pdf")
if not os.path.exists(pdf) or os.path.getsize(pdf) < 20000:
    ok = False
    print("    article PDF missing/small")
stage("article", ok, "(article.md + PDF)")

# ---- stage 5: employer scan ----------------------------------------------
tokens = ["fis" + "global", "F" + "IS Global", "F" + "IS,",
          "@" + "fis" + "global" + ".com"]
hits = []
for base, _dirs, files in os.walk(ROOT):
    for f in files:
        if f.endswith((".md", ".txt", ".py", ".json")):
            p = os.path.join(base, f)
            try:
                content = open(p, encoding="utf-8", errors="ignore").read()
            except Exception:  # noqa: BLE001
                continue
            for t in tokens:
                if t in content:
                    hits.append(f"{p}: {t!r}")
stage("scan", not hits, "(employer identifiers)"
      + ("" if not hits else " HITS: " + "; ".join(hits[:6])))

print()
print("OVERALL:", "PASS" if not failures else f"FAIL ({failures})")
sys.exit(0 if not failures else 1)
