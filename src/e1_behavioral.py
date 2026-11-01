"""E1 - Behavioral one-word swap, expanded.

Operators {discount, increase, of} x rates {5,10,15,20,25,30,40,50}
x final prices {80,120,150,200} = 96 prompts. Greedy decoding, max 48 new
tokens. The stated multiplier is extracted with the documented parser in
common.py and scored against ground truth (1-d, 1+d, d).

Incremental and resumable: results are appended to results/e1_results.json
after every prompt; already-completed prompt keys are skipped on rerun.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
from common import (OPERATORS, PRICES, RATES, chat_prompt, dump_json,  # noqa: E402
                    ground_truth_multiplier, load_model,
                    parse_final_answer, parse_multiplier)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "results", "e1_results.json")


def main():
    tok, model = load_model()
    done = {}
    if os.path.exists(RESULTS):
        import json
        with open(RESULTS) as f:
            for r in json.load(f)["records"]:
                done[r["key"]] = r
    records = []
    combos = [(op, rate, price) for op in OPERATORS for rate in RATES
              for price in PRICES]
    for i, (op, rate, price) in enumerate(combos):
        key = f"{op}|{rate}|{price}"
        if key in done:
            records.append(done[key])
            continue
        prompt = chat_prompt(tok, op, rate, price)
        ids = tok(prompt, return_tensors="pt")
        t0 = time.time()
        # Guard: on the shared host used for these experiments, forward
        # passes intermittently returned non-finite logits. Verify the
        # prompt forward is finite before generating; retry a few
        # times; if it stays non-finite, record the prompt as such
        # (counted incorrect) rather than generating from garbage.
        finite = False
        for _attempt in range(5):
            with torch.no_grad():
                _lg = model(**ids).logits
            if bool(torch.isfinite(_lg.float()).all()):
                finite = True
                break
            time.sleep(2)
        if not finite:
            rec = {
                "key": key, "operator": op, "rate": rate, "price": price,
                "truth_multiplier": ground_truth_multiplier(op, rate),
                "parsed_multiplier": None, "parse_rule": "nonfinite_forward",
                "correct": False, "final_answer_correct": False,
                "generation": "", "seconds": round(time.time() - t0, 2),
            }
            records.append(rec)
            dump_json({"experiment": "E1",
                       "model": "Qwen/Qwen2.5-0.5B-Instruct",
                       "decoding": "greedy, max_new_tokens=48",
                       "n_prompts": len(combos), "records": records},
                      RESULTS)
            print(f"[{i+1}/{len(combos)}] {key} NON-FINITE forward",
                  flush=True)
            continue
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=48, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][ids["input_ids"].shape[1]:],
                         skip_special_tokens=True)
        truth = ground_truth_multiplier(op, rate)
        mult, rule = parse_multiplier(gen, price)
        correct = mult is not None and abs(mult - truth) <= 0.006
        rec = {
            "key": key, "operator": op, "rate": rate, "price": price,
            "truth_multiplier": truth, "parsed_multiplier": mult,
            "parse_rule": rule, "correct": bool(correct),
            "final_answer_correct": bool(parse_final_answer(gen, price, truth)),
            "generation": gen, "seconds": round(time.time() - t0, 2),
        }
        records.append(rec)
        dump_json({"experiment": "E1", "model": "Qwen/Qwen2.5-0.5B-Instruct",
                   "decoding": "greedy, max_new_tokens=48",
                   "n_prompts": len(combos), "records": records}, RESULTS)
        print(f"[{i+1}/{len(combos)}] {key} truth={truth} parsed={mult} "
              f"correct={correct}", flush=True)

    # summary
    summary = {}
    for op in OPERATORS:
        rs = [r for r in records if r["operator"] == op]
        summary[op] = {
            "n": len(rs),
            "multiplier_accuracy": sum(r["correct"] for r in rs) / len(rs),
            "final_answer_accuracy":
                sum(r["final_answer_correct"] for r in rs) / len(rs),
            "unparsed": sum(r["parse_rule"] == "unparsed" for r in rs),
            "by_rate": {
                str(rate): sum(r["correct"] for r in rs if r["rate"] == rate)
                / max(1, sum(1 for r in rs if r["rate"] == rate))
                for rate in RATES
            },
        }
    allacc = sum(r["correct"] for r in records) / len(records)
    dump_json({"experiment": "E1", "model": "Qwen/Qwen2.5-0.5B-Instruct",
               "decoding": "greedy, max_new_tokens=48",
               "n_prompts": len(combos), "overall_multiplier_accuracy": allacc,
               "summary": summary, "records": records}, RESULTS)
    print("SUMMARY", summary, "overall", allacc)


if __name__ == "__main__":
    main()
