# llm-reasoning-demystified

Code, data, figures, and text for the article **"Demystifying LLM
Reasoning: From Language to Mathematical Constructs"** by Ranadhir Ghosh.

The article asks how a language model turns the linguistic expression
"20% discount" into the mathematical object 0.8P, and studies the
question at three levels of evidence: behavioural, representational,
and causal. All experiments run on **Qwen/Qwen2.5-0.5B-Instruct**
(24 layers, hidden size 896), fully locally on CPU in bf16, because its
internals are accessible end to end.

## Contents

- `article/article.md` - the full article text.
- `article/Demystifying_LLM_Reasoning.pdf` - the typeset article.
- `src/` - experiment and figure code (below).
- `results/` - raw outputs of every experiment (JSON), including every
  E1 generation, so all quoted numbers can be recomputed from the
  saved data without re-running the model.
- `figures/` - the six article figures (F1-F6).
- `validate.py` - package integrity harness (see "Validation").

## Experiment map

| ID | Question | Method | Script | Output |
|----|----------|--------|--------|--------|
| E1 | Does the stated multiplier track the operator word? | One-word swap (discount / increase / of) over rates {5..50} x prices {80,120,150,200} = 96 prompts, greedy decoding, documented rule-based parser | `src/e1_behavioral.py` | `results/e1_results.json`, Figure F3 |
| E2 | Are operator identity and multiplier value linearly decodable from hidden states, and where? | Numpy multinomial logistic probes + ridge regression on the final prompt token, per layer, 5-fold CV | `src/e2_probing.py` | `results/e2_results.json`, Figure F4 |
| E3 | Is the operator token causally responsible for the multiplier? | (a) Neutral-word ablation, teacher-forced log-probability drop; (b) residual-stream activation patching between discount and increase runs, per layer | `src/e3_causal.py` | `results/e3_results.json`, Figure F5 |
| E4 | What do attention and the logit lens actually show on the canonical example? | Measured attention from "discount" positions to rate-digit positions (all layers/heads); logit lens at the digit position after "the customer pays " | `src/e4_attention_logitlens.py` | `results/e4_results.json`, Figure F6 |

Shared prompt construction, ground truth, and the E1 parser live in
`src/common.py`. `src/make_diagrams.py` draws the conceptual figures
F1-F2; `src/make_result_figures.py` draws F3-F6 from the results JSONs.
`src/run_all.py` loads the model once and runs E4, E2, E3, E1 in
sequence (useful on slow or shared machines).

Two measurement repairs are part of the recorded pipeline and are
documented where they live: `src/reparse_e1.py` re-extracts E1
multipliers from the saved generations with the final parser (v3:
LaTeX normalisation plus a factored-form rule; the raw generations are
unchanged), and `src/rerun_e3_patching.py` recomputes only the E3
patching block with the corrected first-digit-token scorer (the first
run scored the tokeniser's standalone space token, which made every
gap zero; the ablation block is unchanged).

## How to run

1. Install dependencies: `pip install -r requirements.txt`
   (scikit-learn is not needed; the E2 probes are implemented in
   numpy because the reference environment lacked it).
2. Download the weights of `Qwen/Qwen2.5-0.5B-Instruct` from
   Hugging Face and set `QWEN05B_PATH` to the local directory.
3. From `src/`: run any experiment script, or `python run_all.py`.
   E1 is incremental and resumable; its JSON is rewritten after every
   prompt. On an uncontended machine the full suite takes well under
   an hour; on a heavily shared host it can take much longer.
4. Regenerate figures: `python make_diagrams.py` and
   `python make_result_figures.py` (from `src/`).
5. Check the package: `python validate.py` from the repository root.

## Honest coverage and limitations

- The parser used to score E1 is deterministic and documented in
  `src/common.py`; generations that express the multiplier in an
  unanticipated format are scored "unparsed" and count against
  accuracy. E1 also instructs the model to state the equation in a
  fixed format before solving; it measures multiplier translation
  under a format scaffold, not unaided problem solving.
- The experimental model is small (0.5B parameters). Its failures are
  reported as measured and are part of the article's point; nothing
  here claims the same layer localisation holds in frontier models.
- Probing (E2) establishes linear decodability, not use; the causal
  claims rest on E3. Patching effects are normalised per pair and
  averaged over the documented subset.
- Experiments were run in bf16 on CPU. Logit-lens figures use the
  standard final-norm projection and are approximate by construction.
- The tokeniser splits numbers digit by digit ("20%" is three
  tokens; " 80%" is four, including a leading space token). Position
  level statements in the article and in E4 refer to these tokens.

## Article

The article (PDF in `article/`) develops the conceptual account
(context window, training, transformer mechanics), works the $120 /
20% discount example in full, reports E1-E4 with figures and tables,
presents a clearly labelled pedagogical 3-dimensional walkthrough
alongside its measured counterpart, and discusses where "the equation"
resides in a distributed network. References and a reproducibility
section are included in the article itself.
