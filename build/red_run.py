"""Point red at the rule blue produced and see whether it can get through.

    python3 build/red_run.py --tries 8 --target 60000

Prints every attempt with the referee's verdict on all five constraints, so a failed
evasion is as legible as a successful one. Writes out/red.json for the demo.
"""
import json, os, statistics, sys, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpus as C, facts as F, red as RED, rules as R

# every stage rebuilds the corpus; single-threaded that was 34s each time on a
# 20-core box. Per-account RNG makes the parallel build byte-identical.
import multiprocessing
WORKERS = max(1, multiprocessing.cpu_count() - 2)


OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def arg(f, d, cast=int):
    return cast(sys.argv[sys.argv.index(f) + 1]) if f in sys.argv else d


def main():
    tries = arg("--tries", 8)
    target = arg("--target", 60000)
    rule = json.load(open(os.path.join(OUT, "rule_best.json")))
    rule["gloss"] = R.gloss_from_json(rule)

    c = C.Corpus(seed=7, n_accounts=1200, base_rate=0.01, workers=WORKERS).build()
    wins = c.windows()
    bm = c.branch_map()
    by = {}
    for t in c.txns:
        by.setdefault(t["account_id"], []).append(t)
    amap = dict((a["account_id"], a) for a in c.accounts)

    # a cash-intensive account with real history and no planted case -- red has to do
    # this on a clean front, not on top of someone else's laundering
    cand = [a for a in c.accounts
            if a["cash_intensity"] == "HIGH" and len(by.get(a["account_id"], [])) > 60
            and not any(c.labels["%s-W%02d" % (a["account_id"], w)] for w in range(11))]
    account = cand[0]
    window = [w for w in wins if w["account_id"] == account["account_id"]][-1]
    prior = [t for t in by[account["account_id"]] if t["day"] < window["start"]]
    per_win = {}
    for t in prior:
        if t["direction"] == "in" and t["channel"] == "cash":
            per_win.setdefault(t["day"] // 14, []).append(t["amount"])
    hist = {"avg_dep": int(statistics.mean([len(v) for v in per_win.values()]) if per_win else 0),
            "avg_total": int(statistics.mean([sum(v) for v in per_win.values()]) if per_win else 0),
            "cps": sorted(set(t["counterparty"] for t in prior if t.get("counterparty")))}

    print("RULE UNDER ATTACK")
    print("  " + rule["gloss"].replace("\n", "\n  "))
    print("\nACCOUNT %s -- %s, %s cash, opened %dd ago, branches %s"
          % (account["account_id"], account["sector"], account["cash_intensity"],
             account["opened_days_ago"], ", ".join(account["branches"])))
    print("  normally %d cash deposits per fortnight totalling ~$%s"
          % (hist["avg_dep"], format(hist["avg_total"], ",d")))
    print("\nOBJECTIVE: move $%s in and out inside 14 days without the rule firing.\n"
          % format(target, ",d"))

    def one(i):
        try:
            return RED.attack(rule, account, hist, bm, target, temperature=0.7 + 0.06 * i), None
        except Exception as e:
            return None, str(e)[:70]

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(one, range(tries)))
    print("%d attempts in %.0fs\n" % (tries, time.time() - t0))

    wins_, best = [], None
    for i, (r, err) in enumerate(results, 1):
        if err:
            print("attempt %d  MODEL ERROR: %s" % (i, err))
            continue
        seq = RED.to_txns(account, r.get("transactions") or [], window["start"], bm)
        if not seq:
            print("attempt %d  empty sequence" % i)
            continue
        seq, notes = RED.repair(seq, target)
        ok, d = RED.is_valid_evasion(rule, account, prior, seq, window, target)
        d["repairs"] = notes
        marks = "".join("+" if d[k] else "-" for k in
                        ("moves_enough", "money_left", "within_budget", "plausible", "evades"))
        print("attempt %-2d %s  moved $%-9s end bal $%-8s cost %.2f  %s"
              % (i, marks, format(int(d["moved"]), ",d"), format(int(d["end_balance"]), ",d"),
                 d["cost"], "EVADED" if ok else
                 ("rule fired on %s" % ",".join(d["fired_predicates"]) if d["rule_fired"]
                  else "failed constraints")))
        for nt in d.get("repairs", []):
            print("           referee repaired: %s" % nt)
        if ok:
            wins_.append((r, seq, d))
            if best is None:
                best = (r, seq, d)
    print("\n  legend: moves_enough / money_left / within_budget / plausible / evades")
    print("\nRED SUCCESS RATE: %d of %d (%.0f%%)"
          % (len(wins_), tries, 100.0 * len(wins_) / max(1, tries)))

    if best:
        r, seq, d = best
        print("\n" + "=" * 78)
        print("EVASION FOUND -- verified in Python, no second model call")
        print("=" * 78)
        print("WHY IT GETS THROUGH -- computed from the facts, not from red's account:")
        for line in RED.evasion_lines(rule, d["facts"]):
            print(line)
        print()
        for t in seq:
            print("  day %-3d %-9s $%9s  %s -> %s"
                  % (t["day"] - window["start"] + 1, t["channel"] + " " + t["direction"],
                     format(int(t["amount"]), ",d"), t["branch"], t["counterparty"] or "-"))
        print("\n  moved $%s, end balance $%s, operational cost %.2f"
              % (format(int(d["moved"]), ",d"), format(int(d["end_balance"]), ",d"), d["cost"]))
        print("  fires(rule) = False  <-- the rule does not see it")
        json.dump({"account": account["account_id"], "weakness": r.get("weakness", ""),
                   "why": RED.evasion_lines(rule, d["facts"]),
                   "repairs": d.get("repairs", []),
                   "sequence": seq, "referee": dict((k, v) for k, v in d.items() if k != "facts"),
                   "rule_gloss": rule["gloss"],
                   "success_rate": len(wins_) / float(max(1, tries))},
                  open(os.path.join(OUT, "red.json"), "w"), indent=1)
        print("\nwrote out/red.json")
    else:
        print("\nNo valid evasion in %d attempts. That is a RESULT, not a failure -- it means"
              % tries)
        print("the rule is not trivially evadable at this budget. Report it.")


if __name__ == "__main__":
    main()
