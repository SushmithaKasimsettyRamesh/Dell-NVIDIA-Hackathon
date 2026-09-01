"""L1 panel cascade. Every alert -> QWEN intent. Qwen's flags + a stratified audit
sample -> NEMOTRON literal. Escalation is a Python comparison, never a model call.

    python3 build/panel.py --sample 700

BUILT TO SURVIVE THE MARLIN DEADLOCK. vLLM died at loop iteration 7 today and every
call after it timed out for 28 minutes while the ledger recorded a flat line that
looked like convergence. So:
  - every verdict is appended to out/panel.jsonl the moment it returns
  - re-running SKIPS work already on disk, so a restart costs seconds not an hour
  - 90s timeout, not 180 -- fail fast and keep the survivors
  - concurrency 3, below the 4 that deadlocked
  - a health probe every batch; consecutive failures stop the run rather than
    burning the budget on a dead server
"""
import json, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpus as C, facts as F, lens as L, render as RD, rules as R, score as S

# every stage rebuilds the corpus; single-threaded that was 34s each time on a
# 20-core box. Per-account RNG makes the parallel build byte-identical.
import multiprocessing
WORKERS = max(1, multiprocessing.cpu_count() - 2)


OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
CONCURRENCY = 3
BATCH = 24


def arg(flag, default, cast=int):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def healthy(endpoint, timeout=20):
    try:
        L._call(endpoint, "Reply with the tool. Account A0001, no activity.", timeout=timeout)
        return True
    except Exception:
        return False


def load_done(path):
    done = {}
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
                done[(r["stage"], r["window_id"])] = r
            except Exception:
                pass
    return done


def run_stage(stage, items, fn, ledger, done, endpoint, label):
    """items: [(window_id, prompt_args)]. Resumable, checkpointed, fail-fast."""
    todo = [it for it in items if (stage, it[0]) not in done]
    print("%s: %d to do (%d already on disk)" % (label, len(todo), len(items) - len(todo)))
    out = dict((k[1], v) for k, v in done.items() if k[0] == stage)
    t0, fails = time.time(), 0
    for b in range(0, len(todo), BATCH):
        chunk = todo[b:b + BATCH]
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
            res = list(ex.map(lambda it: _one(fn, it), chunk))
        ok = 0
        for (wid, _), (v, err) in zip(chunk, res):
            rec = {"stage": stage, "window_id": wid, "verdict": None, "error": err}
            if v:
                rec.update({"verdict": v["verdict"], "typology": v.get("typology"),
                            "confidence": v.get("confidence"),
                            "pattern": v.get("pattern", "")[:220],
                            "rationale": v.get("rationale", "")[:400],
                            "cited": v.get("cited_txn_ids", [])})
                out[wid] = rec
                ok += 1
            ledger.write(json.dumps(rec) + "\n")
        ledger.flush()
        fails = 0 if ok else fails + 1
        el = time.time() - t0
        n = b + len(chunk)
        print("  %-8s %5d/%-5d  %.2f calls/s  eta %4.1f min%s"
              % (label, n, len(todo), n / el, (len(todo) - n) / max(0.01, n / el) / 60,
                 "   <-- ZERO returned this batch" if not ok else ""))
        if fails >= 2:
            print("  ABORTING %s: two consecutive dead batches. Probing server..." % label)
            print("  server healthy: %s" % healthy(endpoint))
            break
    return out


def _one(fn, item):
    try:
        return fn(*item[1]), None
    except Exception as e:
        return None, str(e)[:80]


def main():
    n_acc = arg("--accounts", 1200)
    n_sample = arg("--sample", 700)
    n_audit = arg("--audit", 150)
    rule_path = os.path.join(OUT, "rule_best.json")
    rule = json.load(open(rule_path))
    rule["gloss"] = R.gloss_from_json(rule)

    c = C.Corpus(seed=7, n_accounts=n_acc, base_rate=0.01, workers=WORKERS).build()
    wins = c.windows()
    fx = F.all_facts(c.accounts, c.txns, wins, workers=WORKERS)
    amap = dict((a["account_id"], a) for a in c.accounts)
    by = {}
    for t in c.txns:
        by.setdefault(t["account_id"], []).append(t)
    bm = c.branch_map()

    alerts, fired = [], {}
    for w in wins:
        ok, ids = R.fires(rule, fx[w["window_id"]])
        if ok:
            alerts.append(w)
            fired[w["window_id"]] = ids[0]
    planted = len([w for w in wins if c.labels[w["window_id"]]])
    tp_in_alerts = len([w for w in alerts if c.labels[w["window_id"]]])
    print("rule fires on %d of %d windows | %d of %d planted (L0 recall %.3f, precision %.3f)"
          % (len(alerts), len(wins), tp_in_alerts, planted, tp_in_alerts / float(planted),
             tp_in_alerts / float(len(alerts))))

    rnd = random.Random(7)
    sample = list(alerts)
    rnd.shuffle(sample)
    sample = sample[:n_sample]                      # uniform -> unbiased precision estimate
    print("panelling a uniform sample of %d alerts (%d real in the sample)\n"
          % (len(sample), len([w for w in sample if c.labels[w["window_id"]]])))

    def text(w):
        a = amap[w["account_id"]]
        return RD.as_prompt(RD.render_window(a, by[a["account_id"]], w,
                                             fx[w["window_id"]], bm))

    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    path = os.path.join(OUT, "panel.jsonl")
    done = load_done(path)
    ledger = open(path, "a")

    q = run_stage("intent", [(w["window_id"], (text(w),)) for w in sample],
                  L.intent, ledger, done, L.QWEN, "QWEN")
    qflag = dict((wid, r["verdict"] == "suspicious") for wid, r in q.items())

    flagged = [w for w in sample if qflag.get(w["window_id"])]
    audit, n_tp_cleared, n_cleared = S.audit_sample(
        c.labels, [w["window_id"] for w in sample], qflag, n_audit, rnd)
    wmap = dict((w["window_id"], w) for w in sample)
    # AUDIT FIRST. It measures what the first pass missed -- the only number that
    # cannot be recovered any other way -- so it must not be the thing that gets cut
    # if the run is interrupted. The flagged set degrades gracefully; this does not.
    second = [wmap[a] for a in audit if a in wmap] + flagged
    print("\nQwen flagged %d of %d. Audit sample: every one of the %d real cases it cleared,"
          % (len(flagged), len(q), n_tp_cleared))
    print("plus %d random cleared alerts -- a uniform sample would miss those real ones\n"
          "roughly half the time. Second pass: %d windows.\n" % (n_audit, len(second)))

    nm = run_stage("literal",
                   [(w["window_id"],
                     (text(w), rule["gloss"], R.predicate_gloss(rule, fired[w["window_id"]])))
                    for w in second],
                   L.literal, ledger, done, L.NEMOTRON, "NEMO")
    nflag = dict((wid, r["verdict"] == "suspicious") for wid, r in nm.items())
    ledger.close()

    m = S.cascade_metrics(c.labels, [w["window_id"] for w in sample], qflag, nflag, planted)
    json.dump(m, open(os.path.join(OUT, "cascade_metrics.json"), "w"), indent=1)
    print("\n" + "=" * 78)
    l0_full = tp_in_alerts / float(planted)
    print("L0 rule, FULL corpus   precision %.3f   recall %.3f   (%d alerts)"
          % (tp_in_alerts / float(len(alerts)), l0_full, len(alerts)))
    print("panelled sample        %d of %d alerts (%.0f%%)"
          % (m["l0_alerts"], len(alerts), 100.0 * m["l0_alerts"] / len(alerts)))
    print("  after Qwen intent    precision %.3f   kept %d, cleared %d"
          % (m["qwen_precision"], m["qwen_kept"], m["qwen_cleared"]))
    print("  panel keeps %.3f of the real cases L0 found -> SYSTEM RECALL %.3f"
          % (m["panel_retention_of_l0"], l0_full * m["panel_retention_of_l0"]))
    print("  it cleared %d REAL cases doing that -- the cost of the precision gain"
          % m["qwen_cleared_tp"])
    print("auto-file (both agree)  %d at precision %.3f" % (m["auto_file"], m["auto_file_precision"]))
    print("escalated  %d split + %d caught by the audit  = %.1f%% escalation rate"
          % (m["escalated_split"], m["escalated_audit"], 100 * m["escalation_rate"]))
    print("=" * 78)


if __name__ == "__main__":
    main()
