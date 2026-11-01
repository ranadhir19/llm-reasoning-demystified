"""Combined runner: load the model ONCE and run E4, E2, E3, E1 in order
(cheapest first), so each experiment's JSON is on disk before the next
starts. Exists because this shared host has severe IO contention and a
roughly two-hour reboot cadence; a single load amortises the dominant
cost. Each experiment module remains independently runnable.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

torch.set_num_threads(4)
import common  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "results")


def main():
    tok, model = common.load_model()
    print(f"model loaded {time.strftime('%H:%M:%S')}", flush=True)
    shared = (tok, model)
    plan = [("e4_attention_logitlens", "e4_results.json"),
            ("e2_probing", "e2_results.json"),
            ("e3_causal", "e3_results.json"),
            ("e1_behavioral", "e1_results.json")]
    for modname, outfile in plan:
        path = os.path.join(RESULTS_DIR, outfile)
        if os.path.exists(path):
            if outfile == "e1_results.json":
                import json
                with open(path) as f:
                    if "overall_multiplier_accuracy" not in json.load(f):
                        pass  # partial file: E1 resumes internally
                    else:
                        print(f"SKIP {modname} (results exist)", flush=True)
                        continue
            else:
                print(f"SKIP {modname} (results exist)", flush=True)
                continue
        import importlib
        mod = importlib.import_module(modname)
        mod.load_model = lambda: shared
        t0 = time.time()
        print(f"START {modname} {time.strftime('%H:%M:%S')}", flush=True)
        try:
            mod.main()
            print(f"DONE {modname} in {time.time()-t0:.0f}s", flush=True)
        except Exception:  # noqa: BLE001 - one failed experiment must not
            import traceback  # lose the rest of the run
            traceback.print_exc()
            print(f"FAILED {modname} after {time.time()-t0:.0f}s",
                  flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
