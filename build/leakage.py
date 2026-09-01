"""Fingerprint audit -- does any fact separate planted from benign so cleanly that
the generator, not laundering, is the explanation?

This exists because new_branch_count separated at 4.13 sigma with benign at 0.001:
planted windows used unfamiliar branches BY CONSTRUCTION and benign never did. Blue
would have found it in three iterations and the whole demo would have rested on it,
plausibly enough that nobody would notice.

RUN THIS AFTER ANY CHANGE TO THE GENERATOR. Anything over ~2 sigma is suspect.
"""
import os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpus as C, facts as F, rules as R

# every stage rebuilds the corpus; single-threaded that was 34s each time on a
# 20-core box. Per-account RNG makes the parallel build byte-identical.
import multiprocessing
WORKERS = max(1, multiprocessing.cpu_count() - 2)


SUSPECT = 2.0


def derived(w, amap, by):
    """The facts we are considering adding, measured before we add them."""
    a = amap[w["account_id"]]
    txns = by[a["account_id"]]
    inw = [t for t in txns if w["start"] <= t["day"] <= w["end"]]
    prior = [t for t in txns if t["day"] < w["start"]]
    dep = [t["amount"] for t in inw if t["direction"] == "in" and t["channel"] == "cash"]
    outs = [t for t in inw if t["direction"] == "out"]
    pb = set(t["branch"] for t in prior)
    pc = set(t["counterparty"] for t in prior if t.get("counterparty"))
    cv = (statistics.pstdev(dep) / statistics.mean(dep)) if len(dep) > 1 and statistics.mean(dep) else 0.0
    return {
        # candidates only. Anything that graduates into facts.py comes out of here.
        "~outbound_new_cp_value_share": (
            sum(t["amount"] for t in outs if t.get("counterparty") and t["counterparty"] not in pc)
            / max(1.0, sum(t["amount"] for t in outs))) if prior and outs else 0.0,
        "~deposit_gini": cv,
    }


def main():
    n = 400
    if "--accounts" in sys.argv:
        n = int(sys.argv[sys.argv.index("--accounts") + 1])
    c = C.Corpus(seed=7, n_accounts=n, base_rate=0.01, workers=WORKERS).build()
    wins = c.windows()
    fx = F.all_facts(c.accounts, c.txns, wins, workers=WORKERS)
    amap = dict((a["account_id"], a) for a in c.accounts)
    by = {}
    for t in c.txns:
        by.setdefault(t["account_id"], []).append(t)

    alerts = [w for w in wins if R.fires(R.BANK_SCENARIOS, fx[w["window_id"]])[0]]
    pos = [w for w in alerts if c.labels[w["window_id"]]]
    neg = [w for w in alerts if not c.labels[w["window_id"]]]
    print("Alert windows: %d planted, %d benign  (separation measured on ALERTS only --\n"
          "that is the population the panel actually sees)\n" % (len(pos), len(neg)))

    rows = []
    for name in F.FACT_NAMES:
        p = [fx[w["window_id"]][name] for w in pos]
        q = [fx[w["window_id"]][name] for w in neg]
        rows.append((name, p, q))
    dp = [derived(w, amap, by) for w in pos]
    dn = [derived(w, amap, by) for w in neg]
    for k in dp[0]:
        rows.append((k, [d[k] for d in dp], [d[k] for d in dn]))

    out = []
    for name, p, q in rows:
        sd = statistics.pstdev(list(p) + list(q)) or 1e-9
        out.append((abs(statistics.mean(p) - statistics.mean(q)) / sd, name,
                    statistics.mean(p), statistics.mean(q)))
    out.sort(reverse=True)
    print("%-30s %10s %10s %8s" % ("fact", "PLANTED", "BENIGN", "sigma"))
    for sep, name, mp, mn in out[:14]:
        flag = "  <-- SUSPECT" if sep > SUSPECT else ""
        print("%-30s %10.3f %10.3f %8.2f%s" % (name, mp, mn, sep, flag))
    bad = [o for o in out if o[0] > SUSPECT]
    print("\n%d fact(s) over %.1f sigma. %s"
          % (len(bad), SUSPECT,
             "Check the generator before letting blue near them." if bad
             else "No single fact gives the answer away."))
    print("(~ prefixed facts are candidates, not yet in facts.py)")


if __name__ == "__main__":
    main()
