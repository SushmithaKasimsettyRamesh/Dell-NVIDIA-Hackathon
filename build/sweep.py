"""Scale sweep -- score a FIXED rule over a much larger corpus, across all cores.

    python3 build/sweep.py --accounts 10000 --workers 16

Why this exists. The loop and the cascade are GPU-bound and capped at 4 concurrent by
the MARLIN NVFP4 deadlock, so they leave 19 of 20 cores idle. But L0 scoring is pure
Python over a fact dict -- embarrassingly parallel, and read-only with respect to the
rule. So this can run at full width WHILE the GPU is busy, with no contention at all.

It does NOT feed the rule. rule_best.json and the cascade were derived from the
1,200-account corpus and are untouched by anything here. This is a scale measurement:
"the rule we trained on 14,400 windows, applied to 120,000 windows in N seconds."

Each worker builds its own accounts from its own seed, so the corpus differs from a
single-process run by construction. That is fine and intended -- it is a different
sample from the same generator, which is the point of measuring on it.
"""
import json, os, sys, time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpus as C, facts as F, rules as R

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
_RULE = None


def arg(f, d, cast=int):
    return cast(sys.argv[sys.argv.index(f) + 1]) if f in sys.argv else d


def _init(rule):
    global _RULE
    _RULE = rule


def _chunk(job):
    """One worker: build its slice, compute facts, score. Returns counts only --
    shipping 10k accounts of transactions back through a pipe would dominate."""
    seed, n = job
    t0 = time.time()
    c = C.Corpus(seed=seed, n_accounts=n, base_rate=0.01).build()
    wins = c.windows()
    fx = F.all_facts(c.accounts, c.txns, wins)
    tp = fp = fn = tn = 0
    for w in wins:
        wid = w["window_id"]
        fired = R.fires(_RULE, fx[wid])[0]
        real = c.labels[wid] is not None
        if fired and real:
            tp += 1
        elif fired:
            fp += 1
        elif real:
            fn += 1
        else:
            tn += 1
    return {"seed": seed, "accounts": n, "windows": len(wins), "txns": len(c.txns),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "seconds": time.time() - t0}


def main():
    total = arg("--accounts", 10000)
    workers = arg("--workers", 16)
    rule = json.load(open(os.path.join(OUT, "rule_best.json")))
    per = max(50, total // workers)
    jobs = [(1000 + i, per) for i in range(workers)]

    print("scoring the FIXED rule from out/rule_best.json over a fresh corpus")
    print("%d workers x %d accounts = %d accounts\n" % (workers, per, workers * per))
    t0 = time.time()
    with Pool(workers, initializer=_init, initargs=(rule,)) as p:
        res = p.map(_chunk, jobs)
    wall = time.time() - t0

    agg = dict((k, sum(r[k] for r in res)) for k in ("windows", "txns", "tp", "fp", "fn", "tn"))
    alerts = agg["tp"] + agg["fp"]
    planted = agg["tp"] + agg["fn"]
    prec = agg["tp"] / float(alerts) if alerts else 0.0
    rec = agg["tp"] / float(planted) if planted else 0.0
    cpu = sum(r["seconds"] for r in res)

    print("%s windows | %s transactions | %s planted"
          % (format(agg["windows"], ",d"), format(agg["txns"], ",d"), format(planted, ",d")))
    print("rule fires on %s (%.1f%%)  precision %.3f  recall %.3f  FP-rate %.1f%%"
          % (format(alerts, ",d"), 100.0 * alerts / agg["windows"], prec, rec, 100 * (1 - prec)))
    print("\n%.1fs wall clock, %.1fs of CPU across %d cores -- %.1fx speedup, $0, no model calls"
          % (wall, cpu, workers, cpu / wall))
    print("same job single-threaded: %s" % ("~%.1f minutes" % (cpu / 60.0)
                                            if cpu > 60 else "%.0f seconds" % cpu))
    json.dump({"windows": agg["windows"], "txns": agg["txns"], "planted": planted,
               "alerts": alerts, "precision": prec, "recall": rec, "wall_seconds": wall,
               "cpu_seconds": cpu, "workers": workers},
              open(os.path.join(OUT, "sweep.json"), "w"), indent=1)
    print("wrote out/sweep.json")


if __name__ == "__main__":
    main()
