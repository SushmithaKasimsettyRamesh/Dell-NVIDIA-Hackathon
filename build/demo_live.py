"""Live single-case walkthrough. Transactions arrive, the rule fires, two models
judge it, they disagree, a human gets pinged.

    python3 build/demo_live.py                 pick a case the models will split on
    python3 build/demo_live.py --case A0123-W04
    python3 build/demo_live.py --no-telegram   same thing, no phone

Everything here is computed live. The only thing chosen in advance is WHICH account
to show -- the models judge it fresh, on stage, and the escalation is a real comparison
of two verdicts that were not known when the demo started.
"""
import json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpus as C, facts as F, lens as L, render as RD, rules as R

import multiprocessing
WORKERS = max(1, multiprocessing.cpu_count() - 2)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

B, D, R_, G, Y, C_, W, X = ("\033[1m", "\033[2m", "\033[31m", "\033[32m",
                            "\033[33m", "\033[36m", "\033[37m", "\033[0m")


def rule_line(ch="─", n=74):
    print(D + ch * n + X)


def head(t):
    print("\n" + B + t + X)
    rule_line()


def typed(s, delay=0.012):
    for ch in s:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)
    print()


def money(x):
    return "$%s" % format(int(round(x)), ",d")


def main():
    no_tg = "--no-telegram" in sys.argv
    want = sys.argv[sys.argv.index("--case") + 1] if "--case" in sys.argv else None
    slow = 0.0 if "--fast" in sys.argv else 1.0

    print("\033[2J\033[H" + B + "  AML monitoring — live" + X)
    print(D + "  Qwen3.6-35B and Nemotron-3-Nano, both on this box. Nothing leaves it." + X)

    rule = json.load(open(os.path.join(OUT, "rule_best.json")))
    rule["gloss"] = R.gloss_from_json(rule)
    c = C.Corpus(seed=7, n_accounts=1200, base_rate=0.01, workers=WORKERS).build()
    wins = dict((w["window_id"], w) for w in c.windows())
    fx = F.all_facts(c.accounts, c.txns, list(wins.values()), workers=WORKERS)
    amap = dict((a["account_id"], a) for a in c.accounts)
    by = {}
    for t in c.txns:
        by.setdefault(t["account_id"], []).append(t)
    bm = c.branch_map()

    if not want:
        # a case the two models split on last run -- the account is chosen, the verdicts
        # are not. They are recomputed live below.
        prev = {}
        p = os.path.join(OUT, "panel.jsonl")
        if os.path.exists(p):
            for line in open(p):
                r = json.loads(line)
                if r.get("verdict"):
                    prev.setdefault(r["window_id"], {})[r["stage"]] = r["verdict"]
        split = [k for k, v in prev.items()
                 if len(v) == 2 and v.get("intent") != v.get("literal")]
        want = split[0] if split else sorted(wins)[0]

    w = wins[want]
    a = amap[w["account_id"]]
    f = fx[want]
    inw = sorted([t for t in by[a["account_id"]] if w["start"] <= t["day"] <= w["end"]],
                 key=lambda t: (t["day"], t["hour"]))

    head("  ACCOUNT %s — %s, opened %d days ago" % (a["account_id"], a["sector"],
                                                    a["opened_days_ago"]))
    print("  registered branches %s   ·   expected volume %s/month"
          % (", ".join(a["branches"]), money(a["expected_monthly_volume"])))

    head("  TRANSACTIONS ARRIVING")
    run = 0.0
    for t in inw[:16]:
        run += t["amount"] if t["direction"] == "in" else -t["amount"]
        col = G if t["direction"] == "in" else Y
        print("  %sday %-3d %s  %-9s %s%10s%s   %s  %s   %sbalance %s%s"
              % (D, t["day"] - w["start"] + 1, X, t["channel"] + " " + t["direction"],
                 col, money(t["amount"]), X, t["branch"], (t.get("counterparty") or "—")[:10],
                 D, money(run), X))
        time.sleep(0.09 * slow)
    if len(inw) > 16:
        print("  %s… %d more%s" % (D, len(inw) - 16, X))

    head("  RULE EVALUATION")
    print("  %s%s%s" % (D, rule["gloss"].replace("\n", "\n  "), X))
    t0 = time.time()
    fired, which = R.fires(rule, f)
    el = (time.time() - t0) * 1e6
    time.sleep(0.4 * slow)
    if fired:
        print("\n  %sALERT%s — %s" % (R_ + B, X, R.predicate_gloss(rule, which[0])))
    else:
        print("\n  %sno alert%s" % (G, X))
    print("  %sevaluated in %.0f microseconds, in Python, zero model calls%s" % (D, el, X))

    head("  TWO INDEPENDENT REVIEWERS")
    text = RD.as_prompt(RD.render_window(a, by[a["account_id"]], w, f, bm))
    pg = R.predicate_gloss(rule, which[0]) if which else ""

    print("  %sIntent — is there a legitimate purpose?  (Qwen3.6-35B, :8000)%s" % (C_, X))
    t0 = time.time()
    q = L.intent(text)
    print("  %s%s%s  %s(%.1fs)%s" % (R_ + B if q["verdict"] == "suspicious" else G + B,
                                     q["verdict"].upper(), X, D, time.time() - t0, X))
    typed("  " + (q.get("pattern") or q.get("rationale", ""))[:150], 0.008 * slow)

    print("\n  %sLiteral — does the rule's evidence hold?  (Nemotron-3-Nano, :8081)%s" % (C_, X))
    t0 = time.time()
    lit = L.literal(text, rule["gloss"], pg)
    print("  %s%s%s  %s(%.1fs)%s" % (R_ + B if lit["verdict"] == "suspicious" else G + B,
                                     lit["verdict"].upper(), X, D, time.time() - t0, X))
    typed("  " + (lit.get("pattern") or lit.get("rationale", ""))[:150], 0.008 * slow)

    head("  DECISION")
    agree = q["verdict"] == lit["verdict"]
    if agree and q["verdict"] == "suspicious":
        print("  Both reviewers say suspicious → %sdrafted for filing%s" % (R_ + B, X))
    elif agree:
        print("  Both reviewers say benign → %sclosed, no action%s" % (G, X))
    else:
        print("  %s%s%s vs %s%s%s — they disagree." % (C_, q["verdict"], X, C_, lit["verdict"], X))
        print("  %sEscalation is this comparison, in Python. No model decides to escalate.%s"
              % (D, X))
        print("\n  %s%sESCALATING TO A HUMAN%s" % (Y, B, X))
        if not no_tg:
            try:
                import escalate as E
                kind = "split" if q["verdict"] == "suspicious" else "audit"
                E.send(E.card(want, a, f, {"verdict": q["verdict"],
                                           "rationale": q.get("rationale", "")},
                              {"verdict": lit["verdict"],
                               "rationale": lit.get("rationale", "")}, kind))
                print("  %s→ sent to Telegram. Check the phone.%s" % (G, X))
                if "--wait" in sys.argv:
                    E.wait(int(sys.argv[sys.argv.index("--wait") + 1]))
            except Exception as e:
                print("  %stelegram unavailable: %s%s" % (Y, str(e)[:70], X))
    rule_line()
    print()


if __name__ == "__main__":
    main()
