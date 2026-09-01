"""Load the run into MongoDB. Required by the organisers, and the checkpointing we
already needed IS the integration -- no extra work, just a destination.

    python3 build/to_mongo.py

Collections (STACK_INTEGRATION.md): accounts, transactions, facts, decisions, reports,
rules, red. JSONL is still the source of truth for crash recovery -- append is atomic
and survives a kill mid-write, which a database round trip does not. Mongo is the
queryable mirror, written after.
"""
import json, os, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpus as C, facts as F, rules as R

# every stage rebuilds the corpus; single-threaded that was 34s each time on a
# 20-core box. Per-account RNG makes the parallel build byte-identical.
import multiprocessing
WORKERS = max(1, multiprocessing.cpu_count() - 2)


OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
DB = "aml"


def load(coll, path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        print("  %-14s skipped (nothing to load)" % coll)
        return 0
    subprocess.run(["docker", "exec", "mongo", "mongosh", "--quiet", "--eval",
                    "db.%s.drop()" % coll, DB], capture_output=True)
    with open(path) as fh:
        p = subprocess.run(["docker", "exec", "-i", "mongo", "mongoimport",
                            "--quiet", "--db", DB, "--collection", coll], stdin=fh,
                           capture_output=True, text=True)
    n = subprocess.run(["docker", "exec", "mongo", "mongosh", "--quiet", "--eval",
                        "db.%s.countDocuments({})" % coll, DB],
                       capture_output=True, text=True).stdout.strip()
    print("  %-14s %s documents" % (coll, n))
    return n


def main():
    c = C.Corpus(seed=7, n_accounts=1200, base_rate=0.01, workers=WORKERS).build()
    wins = c.windows()
    fx = F.all_facts(c.accounts, c.txns, wins, workers=WORKERS)
    rule = json.load(open(os.path.join(OUT, "rule_best.json")))
    tmp = os.path.join(OUT, "_mongo")
    if not os.path.isdir(tmp):
        os.makedirs(tmp)

    def dump(name, recs):
        p = os.path.join(tmp, name + ".jsonl")
        with open(p, "w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        return p

    print("loading into mongodb://localhost:27017/%s\n" % DB)
    load("accounts", dump("accounts", c.accounts))
    load("transactions", dump("transactions", c.txns))
    load("facts", dump("facts", [dict(f, window_id=w, label=c.labels[w])
                                 for w, f in fx.items()]))

    # every panel verdict, with the ground truth alongside so the run is auditable
    dec = []
    if os.path.exists(os.path.join(OUT, "panel.jsonl")):
        for line in open(os.path.join(OUT, "panel.jsonl")):
            r = json.loads(line)
            r["truth"] = c.labels.get(r.get("window_id"))
            dec.append(r)
    load("decisions", dump("decisions", dec))
    load("reports", os.path.join(OUT, "sars.jsonl"))

    # rule version history as blue forced it to evolve
    hist = []
    if os.path.exists(os.path.join(OUT, "loop.jsonl")):
        for line in open(os.path.join(OUT, "loop.jsonl")):
            r = json.loads(line)
            if r.get("event") in ("iteration", "candidate", "final"):
                hist.append(r)
    load("rules", dump("rules", hist))
    if os.path.exists(os.path.join(OUT, "red.json")):
        load("red", dump("red", [json.load(open(os.path.join(OUT, "red.json")))]))

    print("\nsanity queries:")
    for label, q in (
            ("real cases both models caught",
             "db.decisions.countDocuments({stage:'literal',verdict:'suspicious',truth:{$ne:null}})"),
            ("reports drafted", "db.reports.countDocuments({})"),
            ("reports on planted cases", "db.reports.countDocuments({truth:{$ne:null}})"),
            ("windows the rule fired on", "db.facts.countDocuments({new_branch_count:{$gte:1}})")):
        r = subprocess.run(["docker", "exec", "mongo", "mongosh", "--quiet", "--eval", q, DB],
                           capture_output=True, text=True).stdout.strip()
        print("  %-32s %s" % (label, r))


if __name__ == "__main__":
    main()
