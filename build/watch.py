"""Always-on. The monitor never stops: windows keep arriving, the rule scores every
one, the panel reviews what it flags, and a human gets pinged when the two models split.

    python3 build/watch.py                 run until stopped
    python3 build/watch.py --rate 8        windows per second
    python3 build/watch.py --no-telegram

This is the brief's first word, and the thing most demos fail: nothing runs while
nobody is watching. Leave this in a pane for the whole pitch -- the counters keep
moving while you talk, because the box is genuinely still working.

Concurrency is 1 on purpose. The MARLIN kernel deadlocks above 4 and demo_live.py may
be running beside this; leaving headroom matters more than throughput here.
"""
import json, os, queue, random, subprocess, sys, threading, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpus as C, facts as F, lens as L, render as RD, rules as R

import multiprocessing
WORKERS = max(1, multiprocessing.cpu_count() - 2)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
B, D, RD_, G, Y, C_, X = ("\033[1m", "\033[2m", "\033[31m", "\033[32m",
                          "\033[33m", "\033[36m", "\033[0m")


def arg(f, d, cast=float):
    return cast(sys.argv[sys.argv.index(f) + 1]) if f in sys.argv else d


def hms(s):
    return "%dh%02dm%02ds" % (s // 3600, (s % 3600) // 60, s % 60)


def main():
    rate = arg("--rate", 6.0)
    no_tg = "--no-telegram" in sys.argv
    rule = json.load(open(os.path.join(OUT, "rule_best.json")))
    rule["gloss"] = R.gloss_from_json(rule)

    print("\033[2J\033[H" + B + "  Continuous monitoring — running until stopped" + X)
    print(D + "  loading the book..." + X)
    c = C.Corpus(seed=7, n_accounts=1200, base_rate=0.01, workers=WORKERS).build()
    allw = c.windows()
    fx = F.all_facts(c.accounts, c.txns, allw, workers=WORKERS)
    amap = dict((a["account_id"], a) for a in c.accounts)
    by = {}
    for t in c.txns:
        by.setdefault(t["account_id"], []).append(t)
    bm = c.branch_map()
    rnd = random.Random()

    st = {"scanned": 0, "alerts": 0, "reviewed": 0, "filed": 0, "closed": 0,
          "split": 0, "txns": 0, "calls": 0, "deferred": 0,
          # the corpus is synthetic, so the truth is known -- which is the only reason
          # a live false-positive rate can exist at all. No real bank can compute this.
          "alert_real": 0, "filed_real": 0}
    t0 = time.time()
    ledger = open(os.path.join(OUT, "watch.jsonl"), "a")
    events = []
    state_path = os.path.join(OUT, "live.json")

    def to_mongo(rec):
        """Write each live decision straight into mongo so the database pane MOVES.
        Previously it only changed when to_mongo.py ran at the end of a batch, so it
        sat frozen through the whole demo."""
        try:
            subprocess.run(["docker", "exec", "mongo", "mongosh", "--quiet", "--eval",
                            "db.live.insertOne(%s)" % json.dumps(rec), "aml"],
                           capture_output=True, timeout=8)
        except Exception:
            pass

    def publish(ev=None):
        """The dashboard polls live.json. Written atomically -- the page fetches this
        every second and a half-written file would render as a crash."""
        if ev:
            events.insert(0, ev)
            del events[200:]     # deep enough that the tab counts and the
                                 # visible rows do not diverge during a demo
        el = time.time() - t0
        d = dict(st)
        d.update({
            "uptime": int(el),
            "rule_fp": (100.0 * (1 - st["alert_real"] / float(st["alerts"]))
                        if st["alerts"] else None),
            "panel_fp": (100.0 * (1 - st["filed_real"] / float(st["filed"]))
                         if st["filed"] else None),
            "wps": st["scanned"] / max(1e-6, el),
            "awaiting": min(40, st["alerts"] - st["reviewed"]),
            "deferred": st["deferred"],
            "tps": st["txns"] / max(1e-6, el),
            "events": events,
            "rule": rule["gloss"],
        })
        tmp = state_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(d, fh)
        os.replace(tmp, state_path)

    def status():
        el = time.time() - t0
        rule_fp = 100.0 * (1 - st["alert_real"] / float(st["alerts"])) if st["alerts"] else 0.0
        panel_fp = 100.0 * (1 - st["filed_real"] / float(st["filed"])) if st["filed"] else 0.0
        band = (G if panel_fp < 80 else Y) if st["filed"] else D
        out = ("\033[s\033[3;1H"
               "  %swindows%s %-9s %stransactions%s %-12s %salerts%s %-6s "
               "%sfiled%s %-5s %ssplit%s %-5s %sup%s %s\033[K\n"
               "  %sfalse positives:%s  rule alone %s%.1f%%%s   after both models %s%.1f%%%s"
               "   %s· industry benchmark 95%%%s\033[K\n"
               "  %s%.0f windows/s · %d model calls · %.0f transactions/s · $0%s\033[K"
               "\033[u"
               % (D, X, format(st["scanned"], ",d"), D, X, format(st["txns"], ",d"),
                  D, X, st["alerts"], D, X, st["filed"], D, X, st["split"],
                  D, X, hms(int(el)),
                  D, X, Y + B, rule_fp, X, band + B, panel_fp, X, D, X,
                  D, st["scanned"] / max(1, el), st["calls"],
                  st["txns"] / max(1, el), X))
        sys.stdout.write(out)
        sys.stdout.flush()

    print("\n\n\n\n")       # room for the status block
    print(D + "  " + "-" * 92 + X)
    order = list(allw)
    rnd.shuffle(order)

    # Scoring is free and instant; reviewing is not. Blocking the scan loop on a model
    # call made windows arrive every 3 minutes -- that was a 180s timeout hanging, not
    # slow scanning. Reviews now run on their own thread and the scan never waits.
    TIMEOUT = int(arg("--timeout", 25))
    reviews = queue.Queue(maxsize=40)
    fails = {"streak": 0, "degraded": False}

    def reviewer():
        while True:
            job = reviews.get()
            if job is None:
                return
            wid, a, f, which, truth, stamp = job
            text = RD.as_prompt(RD.render_window(a, by[a["account_id"]],
                                                 wins_by_id[wid], f, bm))
            try:
                q = L.intent(text, timeout=TIMEOUT)
                st["calls"] += 1
                fails["streak"] = 0
                if fails["degraded"]:
                    fails["degraded"] = False
                    print("      %smodels answering again — review resumed%s" % (G, X))
            except Exception as e:
                fails["streak"] += 1
                if fails["streak"] >= 3 and not fails["degraded"]:
                    fails["degraded"] = True
                    print("\n  %s%s models not answering after %d tries — scanning "
                          "continues, review paused.%s" % (Y, B, fails["streak"], X))
                    print("  %sIf the GPU shows ~96%% at ~17W: docker restart "
                          "vllm-primary%s\n" % (D, X))
                reviews.task_done()
                continue
            st["reviewed"] += 1
            if q["verdict"] != "suspicious":
                st["closed"] += 1
                print("      %s└ intent: benign — closed%s" % (G, X))
                publish({"t": stamp, "id": wid, "sector": a["sector"],
                         "amt": int(f["cash_deposit_total"]), "state": "closed",
                         "note": (q.get("pattern") or q.get("rationale", ""))[:120],
                         "real": truth})
                to_mongo({"window_id": wid, "at": stamp, "outcome": "closed",
                          "sector": a["sector"], "amount": int(f["cash_deposit_total"]),
                          "truth": bool(truth)})
                reviews.task_done()
                continue
            try:
                lit = L.literal(text, rule["gloss"],
                                R.predicate_gloss(rule, which[0]) if which else "",
                                timeout=TIMEOUT)
                st["calls"] += 1
            except Exception:
                reviews.task_done()
                continue
            rec = {"window_id": wid, "at": stamp, "intent": q["verdict"],
                   "literal": lit["verdict"]}
            if lit["verdict"] == "suspicious":
                st["filed"] += 1
                if truth:
                    st["filed_real"] += 1
                print("      %s└ both suspicious — report drafted%s" % (RD_, X))
                rec["outcome"] = "filed"
                publish({"t": stamp, "id": wid, "sector": a["sector"],
                         "amt": int(f["cash_deposit_total"]), "state": "filed",
                         "note": (lit.get("pattern") or lit.get("rationale", ""))[:120],
                         "real": truth})
            else:
                st["split"] += 1
                print("      %s└ intent suspicious, literal benign — ESCALATING TO A "
                      "HUMAN%s" % (Y + B, X))
                rec["outcome"] = "split"
                publish({"t": stamp, "id": wid, "sector": a["sector"],
                         "amt": int(f["cash_deposit_total"]), "state": "split",
                         "note": "intent said suspicious, literal said benign",
                         "real": truth})
                if not no_tg:
                    try:
                        import escalate as E
                        E.send(E.card(wid, a, f, q, lit, "split"))
                        print("        %s→ Telegram sent%s" % (G, X))
                    except Exception as e:
                        print("        %stelegram: %s%s" % (Y, str(e)[:50], X))
            rec.update({"sector": a["sector"], "amount": int(f["cash_deposit_total"]),
                        "truth": bool(truth)})
            ledger.write(json.dumps(rec) + "\n")
            ledger.flush()
            to_mongo(rec)
            reviews.task_done()

    wins_by_id = dict((w["window_id"], w) for w in allw)
    threading.Thread(target=reviewer, daemon=True).start()

    i = 0
    try:
        while True:
            w = order[i % len(order)]
            i += 1
            wid = w["window_id"]
            f = fx[wid]
            a = amap[w["account_id"]]
            st["scanned"] += 1
            st["txns"] += len([t for t in by[a["account_id"]]
                               if w["start"] <= t["day"] <= w["end"]])
            fired, which = R.fires(rule, f)
            if st["scanned"] % 5 == 0:
                publish()
            status()
            if not fired:
                time.sleep(1.0 / rate)
                continue

            st["alerts"] += 1
            truth = c.labels.get(wid) is not None
            if truth:
                st["alert_real"] += 1
            stamp = time.strftime("%H:%M:%S")
            print("  %s%s%s  %s%-12s%s %-26s %10s  %sflagged%s"
                  % (D, stamp, X, C_, wid, X, a["sector"][:26],
                     "$" + format(int(f["cash_deposit_total"]), ",d"), Y, X))
            publish({"t": stamp, "id": wid, "sector": a["sector"],
                     "amt": int(f["cash_deposit_total"]), "state": "flagged",
                     "note": R.predicate_gloss(rule, which[0]) if which else "",
                     "real": truth})
            try:
                reviews.put_nowait((wid, a, f, which, truth, stamp))
            except queue.Full:
                # The rule outpaces the panel by ~50x, so a backlog is the normal state,
                # not a fault. Count what we defer rather than dropping it silently --
                # "scanning is free, review is not" is the whole argument for a gate.
                st["deferred"] += 1
            time.sleep(1.0 / rate)
    except KeyboardInterrupt:
        el = time.time() - t0
        print("\n\n  %sstopped after %s%s" % (D, hms(int(el)), X))
        print("  %s windows, %s transactions, %d alerts, %d filed, %d split, %d model calls"
              % (format(st["scanned"], ",d"), format(st["txns"], ",d"),
                 st["alerts"], st["filed"], st["split"], st["calls"]))
        if st["alerts"]:
            print("  false positives: rule alone %.1f%%, after both models %.1f%% "
                  "(industry benchmark 95%%)"
                  % (100.0 * (1 - st["alert_real"] / float(st["alerts"])),
                     100.0 * (1 - st["filed_real"] / float(max(1, st["filed"])))))
        ledger.close()


if __name__ == "__main__":
    main()
