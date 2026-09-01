"""Does the PANEL beat chance on the acceptance set? The human test checks for
leakage; this checks for sufficiency. If Qwen cannot separate these ten, the
cascade adds nothing and the plan needs to change now, not at 16:00."""
import os, sys, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rules as R, render as RD, lens as L, acceptance as A

c, fx, amap, by, items = A.build()
bm = c.branch_map()
gloss = R.gloss_from_json(R.BANK_SCENARIOS)

jobs = []
for i, (w, lab) in enumerate(items, 1):
    a = amap[w["account_id"]]
    text = RD.as_prompt(RD.render_window(a, by[a["account_id"]], w, fx[w["window_id"]], bm))
    pid = R.fires(R.BANK_SCENARIOS, fx[w["window_id"]])[1][0]
    jobs.append((i, lab, text, pid, a["profile"]))

WHICH = sys.argv[1] if len(sys.argv) > 1 else "intent"

def run_intent(j):
    i, lab, text, pid, prof = j
    t0 = time.time()
    try:
        v = (L.intent(text) if WHICH == "intent"
             else L.literal(text, gloss, R.predicate_gloss(R.BANK_SCENARIOS, pid)))
        return (i, lab, prof, v, time.time() - t0, None)
    except Exception as e:
        return (i, lab, prof, None, time.time() - t0, str(e)[:60])

t0 = time.time()
with ThreadPoolExecutor(max_workers=4) as ex:
    res = sorted(ex.map(run_intent, jobs))
wall = time.time() - t0

print("%s lens on the acceptance set" % WHICH.upper(), "-- %d windows in %.1fs (%.2f calls/s)\n" % (len(jobs), wall, len(jobs)/wall))
print("%-4s %-22s %-14s %-11s %-5s %s" % ("WIN", "TRUTH", "VERDICT", "TYPOLOGY", "CONF", "CITED"))
right = 0
for i, lab, prof, v, el, err in res:
    truth = "benign" if lab.startswith("benign") else "suspicious"
    if err:
        print("%-4d %-22s FAILED: %s" % (i, lab, err)); continue
    ok = v["verdict"] == truth
    right += ok
    print("%-4d %-22s %-14s %-11s %-5.2f %d  %s" % (i, lab, v["verdict"], v["typology"],
          v["confidence"], len(v["cited_txn_ids"]), "OK" if ok else "<-- WRONG"))
print("\nSCORE %d/%d   (chance = 5/10)" % (right, len(res)))
for i, lab, prof, v, el, err in res:
    if v and v["verdict"] != ("benign" if lab.startswith("benign") else "suspicious"):
        print("\nWINDOW %d (%s)\n  saw: %s\n  why: %s"
              % (i, lab, v.get("pattern", "")[:200], v["rationale"][:200]))
