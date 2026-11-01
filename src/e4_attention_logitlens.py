"""E4 - Measured attention and logit lens on the canonical example.

Text (plain, no chat template):
  "A shop sells a product for $120 after applying a 20% discount. What was
   its original price? Let the original price be P. A 20% discount means
   the customer pays"

Tokenisation facts for this model (verified): "20%" is three tokens
('2', '0', '%'); " discount" is a single token; the continuation " 80%"
is four tokens (' ', '8', '0', '%'). Positions are located by character
offsets of the exact substrings.

(a) Attention: attention weights from each "discount" token position to
    the rate-digit token positions of each genuine "20%" occurrence (the
    "20" inside "$120" is excluded), all layers/heads; report the maximum
    and a top-10 table, plus the focused pair (scratchpad "discount" ->
    the "20" of its own sentence). This is the measured counterpart of
    the hypothetical attention matrix in the pedagogical walkthrough.

(b) Logit lens: append the space token ("...customer pays" + " ") so the
    next-token decision is the first digit of the percentage. At that
    position, project every layer's hidden state through the final
    RMSNorm and the (tied) lm_head; record P('8') (start of "80") and
    P('2') (start of "20") per layer. Also record the teacher-forced
    log-probability of the full continuations " 80%" and " 20%" after
    the original text.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from common import dump_json, load_model  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "results", "e4_results.json")

TEXT = ("A shop sells a product for $120 after applying a 20% discount. "
        "What was its original price? Let the original price be P. "
        "A 20% discount means the customer pays")


def char_span_positions(tok, text, needle, skip_first_of_120=False):
    """Token indices whose character span overlaps each occurrence of
    needle. Returns list of token-index lists, one per occurrence."""
    enc = tok(text, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    out = []
    start = 0
    while True:
        i = text.find(needle, start)
        if i < 0:
            break
        end = i + len(needle)
        out.append([j for j, (a, b) in enumerate(offs) if a < end and b > i])
        start = end
    return out


def seq_logprob(tok, model, prompt, cont):
    pids = tok(prompt, return_tensors="pt")["input_ids"]
    cids = tok(cont, add_special_tokens=False, return_tensors="pt")["input_ids"]
    ids = torch.cat([pids, cids], dim=1)
    with torch.no_grad():
        logits = model(ids).logits[0]
    lp = torch.log_softmax(logits.float(), dim=-1)
    start = pids.shape[1]
    return float(sum(lp[start + i - 1, cids[0, i]].item()
                     for i in range(cids.shape[1])))


def main():
    tok, model = load_model()
    ids = tok(TEXT, return_tensors="pt")
    with torch.no_grad():
        out = model(**ids, output_attentions=True, output_hidden_states=True)
    tokens = [tok.decode([i]) for i in ids["input_ids"][0].tolist()]
    disc_occ = char_span_positions(tok, TEXT, "discount")
    rate_all = char_span_positions(tok, TEXT, "20")
    # rate_all[0] is the "20" inside "$120"; genuine rate occurrences:
    rate_occ = rate_all[1:]
    print("discount occurrences:", disc_occ, "rate occurrences:", rate_occ)

    # ---- (a) attention discount -> rate digits ----
    entries = []
    for li, att in enumerate(out.attentions):
        A = att[0].float().numpy()  # (heads, seq, seq)
        for oi, dpos_list in enumerate(disc_occ):
            for ri, rpos_list in enumerate(rate_occ):
                for dp in dpos_list:
                    for rp in rpos_list:
                        for h in range(A.shape[0]):
                            entries.append({
                                "layer": li, "head": h,
                                "discount_occurrence": oi + 1,
                                "rate_occurrence": ri + 1,
                                "discount_pos": dp, "rate_pos": rp,
                                "weight": float(A[h, dp, rp]),
                            })
    entries.sort(key=lambda e: -e["weight"])
    top = entries[:10]
    focused = [e for e in entries if e["discount_occurrence"] == 2
               and e["rate_occurrence"] == 2]
    focused.sort(key=lambda e: -e["weight"])
    problem_pair = [e for e in entries if e["discount_occurrence"] == 1
                    and e["rate_occurrence"] == 1]
    problem_pair.sort(key=lambda e: -e["weight"])

    # ---- (b) logit lens at the digit position ----
    text2 = TEXT + " "
    ids2 = tok(text2, return_tensors="pt")
    with torch.no_grad():
        out2 = model(**ids2, output_hidden_states=True)
    norm = model.model.norm
    head = model.lm_head
    id8 = tok("8", add_special_tokens=False)["input_ids"][0]
    id2 = tok("2", add_special_tokens=False)["input_ids"][0]
    lens = {"layers": [], "p_digit8": [], "p_digit2": [], "rank_digit8": []}
    for L, h in enumerate(out2.hidden_states):
        with torch.no_grad():
            logits = head(norm(h[0, -1])).float()
        probs = torch.softmax(logits, dim=-1)
        lens["layers"].append(L)
        lens["p_digit8"].append(float(probs[id8]))
        lens["p_digit2"].append(float(probs[id2]))
        lens["rank_digit8"].append(
            int((probs > probs[id8]).sum().item()) + 1)
    final_logits = out2.logits[0, -1]
    top_next = torch.topk(torch.softmax(final_logits.float(), dim=-1), 5)
    top_next_list = [{"token": tok.decode([int(i)]), "prob": float(p)}
                     for p, i in zip(top_next.values, top_next.indices)]
    if not (np.isfinite(lens["p_digit8"]).all()
            and np.isfinite(lens["p_digit2"]).all()):
        # Intermittent non-finite forwards were observed on the shared
        # host; fail loudly so the supervisor retries in a fresh
        # process rather than recording a poisoned lens.
        raise RuntimeError("non-finite logit lens in E4")
    lp80 = seq_logprob(tok, model, TEXT, " 80%")
    lp20 = seq_logprob(tok, model, TEXT, " 20%")

    dump_json({
        "experiment": "E4", "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "text": TEXT, "tokens": tokens,
        "discount_occurrences": disc_occ, "rate_occurrences": rate_occ,
        "attention_top10": top,
        "attention_focused_scratchpad_pair_top5": focused[:5],
        "attention_problem_pair_top5": problem_pair[:5],
        "attention_max": top[0],
        "logit_lens": lens,
        "logit_lens_position": "after trailing space token; next-token "
                               "candidates '8' (id %d) vs '2' (id %d)"
                               % (id8, id2),
        "digit_position_top5_next_tokens": top_next_list,
        "continuation_logprob_80pct": lp80,
        "continuation_logprob_20pct": lp20,
    }, RESULTS)
    print("ATTN MAX:", top[0])
    print("FOCUSED:", focused[:3])
    print("LENS p8:", [round(x, 4) for x in lens["p_digit8"]])
    print("DIGIT TOP:", top_next_list)
    print("LP 80% vs 20%:", lp80, lp20)


if __name__ == "__main__":
    main()
