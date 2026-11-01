"""Re-parse saved E1 generations with parser v2 (LaTeX normalisation).

The generations in results/e1_results.json are the raw experimental data
and are NOT modified. Only the extraction fields (parsed_multiplier,
parse_rule, correct, final_answer_correct) and the summary block are
recomputed, using the same summary schema as e1_behavioral.py. Run after
e1_behavioral.py has finished collecting.
"""
import json
import os

from common import (OPERATORS, RATES, parse_final_answer, parse_multiplier)

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "..", "results", "e1_results.json")


def main():
    with open(PATH) as f:
        data = json.load(f)
    records = data["records"]
    for r in records:
        if r.get("parse_rule") == "nonfinite_forward":
            continue
        mult, rule = parse_multiplier(r["generation"], r["price"])
        r["parsed_multiplier"] = mult
        r["parse_rule"] = rule
        r["correct"] = bool(mult is not None
                            and abs(mult - r["truth_multiplier"]) <= 0.006)
        r["final_answer_correct"] = bool(
            parse_final_answer(r["generation"], r["price"],
                               r["truth_multiplier"]))
    summary = {}
    for op in OPERATORS:
        rs = [r for r in records if r["operator"] == op]
        summary[op] = {
            "n": len(rs),
            "multiplier_accuracy":
                sum(r["correct"] for r in rs) / max(1, len(rs)),
            "final_answer_accuracy":
                sum(r["final_answer_correct"] for r in rs) / max(1, len(rs)),
            "unparsed": sum(1 for r in rs if r["parse_rule"] is None),
            "by_rate": {
                str(rate):
                    sum(r["correct"] for r in rs if r["rate"] == rate)
                    / max(1, sum(1 for r in rs if r["rate"] == rate))
                for rate in RATES
            },
        }
    allacc = sum(r["correct"] for r in records) / max(1, len(records))
    data.update({
        "overall_multiplier_accuracy": allacc,
        "summary": summary,
        "parser_version": "v3 (LaTeX-normalising, factored-form rule); "
                          "saved generations re-parsed offline, raw "
                          "generation text unchanged",
    })
    with open(PATH, "w") as f:
        json.dump(data, f, indent=1)
    print("reparsed", len(records), "records; overall", round(allacc, 4),
          {op: round(s["multiplier_accuracy"], 3)
           for op, s in summary.items()})


if __name__ == "__main__":
    main()
