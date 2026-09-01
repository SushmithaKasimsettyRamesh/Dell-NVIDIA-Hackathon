"""The rewrite loop. Blue proposes, fires() scores, best-of-k survives.

    python3 build/loop.py --once              one iteration, k=4, prints everything
    python3 build/loop.py --iters 100 --k 4   the real run

Scoring is deterministic Python over every dev window -- so dev-set SIZE IS FREE and
there is no reason to shrink it. The only model calls are the k rewrites per iteration.

Checkpoints append to out/loop.jsonl every iteration, including REJECTED candidates:
the loop keeps only F1 improvements, so the precision/recall trade-off is invisible on
the accepted path. The rejected ones are where "it caught more, and flagged every
payroll company doing it" actually shows up.
"""
import json, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blue as B, corpus as C, facts as F, rules as R, score as S

# every stage rebuilds the corpus; single-threaded that was 34s each time on a
# 20-core box. Per-account RNG makes the parallel build byte-identical.
import multiprocessing
WORKERS = max(1, multiprocessing.cpu_count() - 2)


OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
MAX_CONCURRENCY = 4        # NVFP4 MoE kernel deadlocks at 8 -- BOX_STATE.md


def arg(flag, default, cast=int):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def split(accounts, wins, seed=7):
    """Split by ACCOUNT, not by window. The same account in both halves leaks its own
    history, its sector and its branch habits across the boundary."""
    ids = sorted(set(a["account_id"] for a in accounts))
    random.Random(seed).shuffle(ids)
    dev_ids = set(ids[:len(ids) // 2])
    dev = [w["window_id"] for w in wins if w["account_id"] in dev_ids]
    hold = [w["window_id"] for w in wins if w["account_id"] not in dev_ids]
    return dev, hold


OBJ = "f2"
BUDGET = 0.15


def obj(m, n):
    return S.objective(m, OBJ, n_windows=n, budget=BUDGET)


def evaluate(rule, fx, window_ids, labels):
    pred = dict((w, R.fires(rule, fx[w])[0]) for w in window_ids)
    m = S.metrics(pred, dict((w, labels[w]) for w in window_ids))
    fns = [w for w in window_ids if labels[w] and not pred[w]]
    fps = [w for w in window_ids if not labels[w] and pred[w]]
    return m, fns, fps


def main():
    global OBJ, BUDGET
    OBJ = sys.argv[sys.argv.index("--objective") + 1] if "--objective" in sys.argv else "f2"
    BUDGET = arg("--budget", 0.15, float)
    iters = arg("--iters", 1 if "--once" in sys.argv else 100)
    k = arg("--k", 4)
    n_acc = arg("--accounts", 400)
    t0 = time.time()

    c = C.Corpus(seed=7, n_accounts=n_acc, base_rate=0.01, workers=WORKERS).build()
    wins = c.windows()
    fx = F.all_facts(c.accounts, c.txns, wins, workers=WORKERS)
    labels = c.labels
    dev, hold = split(c.accounts, wins, seed=7)
    fx_sample = [fx[w] for w in dev[:40]]

    rule = dict(R.SEED_RULE)
    rule["gloss"] = R.gloss_from_json(rule)
    best_m, fns, fps = evaluate(rule, fx, dev, labels)
    h0 = evaluate(rule, fx, hold, labels)[0]

    print("corpus %d accounts, %d windows in %.1fs" % (n_acc, len(wins), time.time() - t0))
    print("dev %d windows (%d real)   holdout %d windows (%d real)"
          % (len(dev), best_m["positives"], len(hold), h0["positives"]))
    print("objective: %s%s" % (OBJ, "  budget %.0f%% of windows" % (100 * BUDGET)
                               if OBJ == "budget" else ""))
    print("seed rule   dev %s %.4f  F1 %.4f (P %.3f R %.3f)   holdout F1 %.4f\n"
          % (OBJ, obj(best_m, len(dev)), best_m["f1"], best_m["precision"],
             best_m["recall"], h0["f1"]))

    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    ledger = open(os.path.join(OUT, "loop.jsonl"), "a")

    def record(**kw):
        ledger.write(json.dumps(kw) + "\n")
        ledger.flush()

    record(event="start", iters=iters, k=k, dev=len(dev), holdout=len(hold),
           dev_positives=best_m["positives"], seed_f1=best_m["f1"],
           seed_holdout_f1=h0["f1"], rule=rule["predicates"])

    n_bad = 0
    for i in range(1, iters + 1):
        prompt = B.build_prompt(rule, fx, dev, labels, best_m, fns, fps)
        t1 = time.time()
        with ThreadPoolExecutor(max_workers=min(k, MAX_CONCURRENCY)) as ex:
            raw = list(ex.map(lambda _: _safe(prompt), range(k)))
        el = time.time() - t1

        cands = []
        for r, err in raw:
            if err:
                n_bad += 1
                record(event="rejected", i=i, reason=err)
                continue
            cand, why = B.validate(r, fx_sample)
            if cand is None:
                n_bad += 1
                record(event="rejected", i=i, reason=why, raw=r)
                continue
            m, _, _ = evaluate(cand, fx, dev, labels)
            cands.append((m, cand))
            # every candidate, accepted or not -- this is where the trade-off lives
            record(event="candidate", i=i, f1=m["f1"], precision=m["precision"],
                   recall=m["recall"], alerts=m["alerts"], change=cand.get("change", ""),
                   rule=cand["predicates"])

        improved = ""
        if cands:
            cands.sort(key=lambda x: -obj(x[0], len(dev)))
            m, cand = cands[0]
            if obj(m, len(dev)) > obj(best_m, len(dev)):
                rule, best_m = cand, m
                best_m, fns, fps = evaluate(rule, fx, dev, labels)
                improved = "  <-- kept"
        hm = None
        if i % 10 == 0 or i == iters:
            # display only. NEVER used for selection -- that would make it a dev set.
            hm = evaluate(rule, fx, hold, labels)[0]
        record(event="iteration", i=i, best_f1=best_m["f1"], precision=best_m["precision"],
               recall=best_m["recall"], alerts=best_m["alerts"], n_valid=len(cands),
               holdout_f1=(hm["f1"] if hm else None), seconds=el, rule=rule["predicates"])
        print("iter %3d  %s %.4f  F1 %.4f  P %.3f R %.3f  alerts %5d  %d/%d valid  %4.1fs%s%s"
              % (i, OBJ, obj(best_m, len(dev)), best_m["f1"], best_m["precision"],
                 best_m["recall"], best_m["alerts"],
                 len(cands), k, el, "  holdout %.4f" % hm["f1"] if hm else "", improved))

    final = evaluate(rule, fx, hold, labels)[0]
    record(event="final", holdout=final, dev=best_m, rule=rule["predicates"],
           gloss=R.gloss_from_json(rule), rejected=n_bad)
    print("\nFINAL  dev F1 %.4f -> holdout F1 %.4f  (P %.3f R %.3f on holdout)"
          % (best_m["f1"], final["f1"], final["precision"], final["recall"]))
    print("seed holdout F1 %.4f -> %.4f    %d invalid candidates discarded"
          % (h0["f1"], final["f1"], n_bad))
    print("\n" + R.gloss_from_json(rule))
    json.dump(rule, open(os.path.join(OUT, "rule_final.json"), "w"), indent=1)
    ledger.close()


def _safe(prompt):
    try:
        return B.propose(prompt), None
    except Exception as e:
        return None, str(e)[:80]


if __name__ == "__main__":
    main()
