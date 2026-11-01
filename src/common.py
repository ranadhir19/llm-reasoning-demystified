"""Shared utilities for the llm-reasoning-demystified experiments.

Model: Qwen/Qwen2.5-0.5B-Instruct, loaded from a local directory (offline).
All experiments use plain-text or chat-templated prompts built from one
template family so the operator word is the only semantic difference.
"""
import json
import os
import re

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = os.environ.get(
    "QWEN05B_PATH",
    os.path.expanduser(
        "~/workspace/harness-ablation-lab/models/qwen2.5-0.5b-instruct"
    ),
)
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

OPERATORS = ["discount", "increase", "of"]
RATES = [5, 10, 15, 20, 25, 30, 40, 50]
PRICES = [80, 120, 150, 200]


def ground_truth_multiplier(operator, rate):
    d = rate / 100.0
    if operator == "discount":
        return 1.0 - d
    if operator == "increase":
        return 1.0 + d
    if operator == "of":
        return d
    raise ValueError(operator)


def problem_text(operator, rate, price):
    """The bare problem statement (no instructions)."""
    if operator == "discount":
        return (f"A shop sells a product for ${price} after applying a "
                f"{rate}% discount. What was its original price?")
    if operator == "increase":
        return (f"A shop sells a product for ${price} after applying a "
                f"{rate}% increase. What was its original price?")
    if operator == "of":
        return (f"A shop sells a product for ${price}, which is {rate}% "
                f"of its original price. What was its original price?")
    raise ValueError(operator)


def chat_prompt(tok, operator, rate, price):
    """Chat-templated prompt used for generation (E1) and probing (E2)."""
    user = (
        problem_text(operator, rate, price)
        + " Let P be the original price. First write the equation in the "
          "form PRICE = MULTIPLIER * P (for example: 120 = 0.8 * P), then "
          "solve for P. Be concise."
    )
    msgs = [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def load_model():
    """Load the model with weights resident in anonymous memory.

    Two failure modes on the heavily shared host used for these
    experiments motivated this loader:
    (1) from_pretrained's mmap-backed safetensors weights were evicted
        from the page cache between forward passes, forcing the 988 MB
        to be re-faulted from disk (~1 MB/s effective; 6.2 GB of reads
        measured before the first experiment finished).
    (2) Loading via from_pretrained and then cloning every tensor made
        the mmap pages and the clones coexist, peaking over the
        available RAM and getting the process OOM-killed.
    Building the model on the meta device and assigning a state dict
    parsed from the raw file bytes avoids both: no mmap, no duplicate
    copy. The on-disk weights are bf16; an fp32 upcast would double
    the footprint to ~2 GB, which this host cannot spare.
    """
    from safetensors import safe_open
    from transformers import AutoConfig

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    print("loader: tokenizer ok", flush=True)
    cfg = AutoConfig.from_pretrained(MODEL_PATH)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            cfg, torch_dtype=torch.bfloat16, attn_implementation="eager")
    print("loader: meta model ok", flush=True)
    # Stream the checkpoint tensor by tensor, cloning each into the
    # model. (Reading the whole file into bytes and parsing it at once
    # made the raw bytes and the parsed tensors coexist - a ~2.3 GB
    # peak that got the process OOM-killed on this shared host; the
    # mmap-backed from_pretrained load was separately unusable because
    # its pages were evicted between forward passes.)
    modules = dict(model.named_modules())
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    with safe_open(os.path.join(MODEL_PATH, "model.safetensors"),
                   framework="pt") as f:
        for key in f.keys():
            t = f.get_tensor(key).clone()
            owner_name, _, attr = key.rpartition(".")
            owner = modules[owner_name] if owner_name else model
            if key in params:
                # a meta parameter cannot take a real tensor via
                # .data (set_data type check); replace the Parameter
                setattr(owner, attr, torch.nn.Parameter(
                    t, requires_grad=params[key].requires_grad))
            elif key in buffers:
                owner._buffers[attr] = t
            else:
                setattr(owner, attr, torch.nn.Parameter(t))
    print("loader: weights streamed in", flush=True)
    # The checkpoint omits lm_head.weight (tied embeddings);
    # tie_weights() binds it to embed_tokens.weight.
    model.tie_weights()
    # The rotary-embedding inv_freq buffer is not in the checkpoint
    # (persistent=False) and is left on the meta device by the
    # construction above; materialise it explicitly with the standard
    # RoPE initialisation so no meta tensor remains in the model.
    rot = model.model.rotary_emb
    if rot.inv_freq.device.type == "meta":
        head_dim = (model.config.hidden_size
                    // model.config.num_attention_heads)
        base = float(getattr(model.config, "rope_theta", 1000000.0))
        inv_freq = 1.0 / (base ** (torch.arange(
            0, head_dim, 2, dtype=torch.float32) / head_dim))
        rot.inv_freq = inv_freq
        if hasattr(rot, "original_inv_freq"):
            rot.original_inv_freq = inv_freq
    model.eval()
    torch.set_num_threads(4)
    print("loader: weights assigned", flush=True)
    return tok, model


# --------------------------------------------------------------------------
# Parser for E1 generations.
#
# Goal: extract the multiplier the model committed to, i.e. the number m in
# an equation of the form PRICE = m * P (or an equivalent statement such as
# "the customer pays 80%", "P = PRICE / m", "multiplier is 0.8").
# Rules are applied in order; the first match wins. All rules are purely
# textual and deterministic. Tolerance for "correct" is |m - truth| <= 0.006.
# --------------------------------------------------------------------------

_NUM = r"(\d+(?:\.\d+)?)"


def parse_multiplier(text, price):
    """Return (multiplier or None, rule_name).

    Parser v2 (2026-11-01): the model's generations render equations in
    LaTeX (e.g. "\\[ 80 = 0.8 \\times P \\]"), which parser v1 missed
    entirely. v2 normalises LaTeX delimiters and operators before applying
    the same rules in the same order. v3 adds Rule 8 for the factored form
    "P * (1 + 0.05)", which v2 left unparsed. Generations are unchanged;
    only the extraction step was repaired, and E1 records were re-parsed
    offline with this version (see src/reparse_e1.py).
    """
    t = text.replace(",", "")
    t = (t.replace("\\times", "*").replace("\\cdot", "*")
           .replace("\\[", " ").replace("\\]", " ")
           .replace("\\(", " ").replace("\\)", " ").replace("$", " "))
    # Rule 1: explicit equation PRICE = m * P  (also PRICE = mP, PRICE = m x P)
    m = re.search(rf"{price}\s*=\s*{_NUM}\s*(?:\*|x|\u00d7)?\s*P\b", t)
    if m:
        v = float(m.group(1))
        return _fix_percent(v, t, m.start(1)), "eq_price_eq_mP"
    # Rule 2: P = PRICE / m
    m = re.search(rf"P\s*=\s*{price}\s*/\s*{_NUM}", t)
    if m:
        return float(m.group(1)), "P_eq_price_over_m"
    # Rule 3: m * P = PRICE  or  mP = PRICE
    m = re.search(rf"{_NUM}\s*(?:\*|x|\u00d7)?\s*P\s*=\s*{price}", t)
    if m:
        v = float(m.group(1))
        return _fix_percent(v, t, m.start(1)), "mP_eq_price"
    # Rule 4: stated percentage the customer pays / that remains:
    # "pays 80%", "pay 80%", "is 80% of", "80% of the original"
    m = re.search(rf"(?:pays?|paying|paid|remains?|remaining|is|equals?)\s+"
                  rf"{_NUM}\s*%\s*of", t)
    if m:
        return float(m.group(1)) / 100.0, "stated_percent_of"
    # Rule 5: "multiplier (is|of) m" / "multiply by m" / "times m"
    m = re.search(rf"(?:multiplier(?:\s+is|\s+of)?|multiply(?:\s+by)?|times)\s+"
                  rf"{_NUM}", t)
    if m:
        v = float(m.group(1))
        return (v / 100.0 if v > 2.5 else v), "multiplier_word"
    # Rule 6: any decimal d in (0, 2) immediately tied to P: "0.8P" / "0.8 P"
    m = re.search(rf"{_NUM}\s*\*?\s*P\b", t)
    if m:
        v = float(m.group(1))
        if 0 < v <= 2.5:
            return v, "decimal_before_P"
    # Rule 7: m as a percentage directly modifying P: "80% of P" handled by 4;
    # "80%P" rare
    m = re.search(rf"{_NUM}\s*%\s*(?:of\s+)?P\b", t)
    if m:
        return float(m.group(1)) / 100.0, "percent_before_P"
    # Rule 8 (v3): factored form "P * (1 + 0.05) = PRICE" or
    # "(1 - 0.20) * P": the multiplier is stated as 1 +/- x.
    m = re.search(rf"\(\s*1\s*([+\-])\s*{_NUM}\s*\)", t)
    if m:
        x = float(m.group(2))
        return (1.0 + x) if m.group(1) == "+" else (1.0 - x), \
            "factored_1_pm_x"
    return None, "unparsed"


def _fix_percent(v, text, pos):
    """If the number captured in an equation is followed by '%', it is a
    percentage, not a decimal multiplier."""
    tail = text[pos:pos + 12]
    m = re.match(r"\d+(?:\.\d+)?\s*%", tail)
    if m:
        return v / 100.0
    return v


def parse_final_answer(text, price, truth_mult):
    """Secondary check: does the generation state the correct original price
    P = price / truth_mult? Looks for 'P = <num>' or 'original price is <num>'
    where <num> is a computed value (not price/mult restated)."""
    true_p = price / truth_mult
    t = text.replace(",", "").replace("$", "")
    cands = []
    for m in re.finditer(r"P\s*=\s*(\d+(?:\.\d+)?)", t):
        cands.append(float(m.group(1)))
    for m in re.finditer(r"original price (?:is|was)\s*(?:about\s*)?(\d+(?:\.\d+)?)", t):
        cands.append(float(m.group(1)))
    for c in cands:
        if abs(c - true_p) <= max(1.0, 0.01 * true_p):
            return True
    return False


def dump_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)
