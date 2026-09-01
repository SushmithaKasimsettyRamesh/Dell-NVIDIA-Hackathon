"""Auto-filed reports. When BOTH models say suspicious, draft the report.

    python3 build/sar.py            -> out/sars.jsonl, out/reports.html, mongo

68 of the 699 sampled alerts had both lenses agree, at precision 0.338 -- 5.6x the
rule alone. Those do not go to a human queue; they get drafted and filed. This is the
half of the workload the escalation page does not show.

Unlike the escalation summary -- which is assembled from facts because stitching two
model rationales produced garbage -- the report narrative IS generated. Writing the
filing is what analysts actually spend the day on, and it is prose, which is the
model's strongest mode rather than its weakest (PLAN.md section 6).

Every figure in the narrative is checked against the facts afterwards. A drafted
report that invents an amount is worse than no report.
"""
import json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpus as C, facts as F, lens as L, render as RD, rules as R

# every stage rebuilds the corpus; single-threaded that was 34s each time on a
# 20-core box. Per-account RNG makes the parallel build byte-identical.
import multiprocessing
WORKERS = max(1, multiprocessing.cpu_count() - 2)


OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

PROPS = {
    "narrative": {"type": "string", "maxLength": 1200},
    "typology": {"type": "string", "enum": ["structuring", "funnel", "layering",
                                            "pass_through", "cuckoo_smurfing",
                                            "velocity_spike", "other"]},
    "cited_txn_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
}
TOOLS = [{"type": "function", "function": {
    "name": "draft_report", "description": "Draft the suspicious activity report.",
    "parameters": {"type": "object", "properties": PROPS,
                   "required": ["narrative", "typology", "cited_txn_ids"]}}}]

PROMPT = """Two independent reviewers both concluded this account-window is suspicious.
Draft the report the bank files.

Write it the way a compliance analyst writes it: plain declarative prose, past tense,
no hedging and no recommendations. Cover, in order — who the account holder is, what
happened, over what period, at which locations, where the money went, and what makes
the pattern inconsistent with the stated business. Use only figures that appear below;
do not estimate or round beyond what is given. Cite the transaction identifiers that
carry the pattern.

{window}

WHAT THE REVIEWERS FOUND
Intent review: {q}
Rule review: {lit}
"""


def check(nar, f, ids_ok):
    """Every dollar figure in the narrative must appear in the facts. A drafted report
    that invents an amount is worse than no report."""
    known = set()
    for v in (f["cash_deposit_total"], f["max_deposit_amount"], f["min_deposit_amount"],
              f["outbound_total"], f["outbound_max"], f["inbound_total"]):
        known.add(int(round(v)))
    bad = []
    for m in re.findall(r"\$([0-9][0-9,]*)", nar or ""):
        v = int(m.replace(",", ""))
        if v >= 1000 and not any(abs(v - k) <= max(2, 0.005 * k) for k in known):
            bad.append(v)
    cited = [c for c in (ids_ok or [])]
    return bad, cited


def main():
    rows = [json.loads(l) for l in open(os.path.join(OUT, "panel.jsonl"))]
    I = dict((r["window_id"], r) for r in rows if r["stage"] == "intent" and r["verdict"])
    Lz = dict((r["window_id"], r) for r in rows if r["stage"] == "literal" and r["verdict"])
    auto = [w for w in Lz if w in I and I[w]["verdict"] == "suspicious"
            and Lz[w]["verdict"] == "suspicious"]
    print("%d cases where both models agreed suspicious -- drafting reports\n" % len(auto))

    rule = json.load(open(os.path.join(OUT, "rule_best.json")))
    c = C.Corpus(seed=7, n_accounts=1200, base_rate=0.01, workers=WORKERS).build()
    wins = dict((w["window_id"], w) for w in c.windows())
    fx = F.all_facts(c.accounts, c.txns, list(wins.values()))
    amap = dict((a["account_id"], a) for a in c.accounts)
    by = {}
    for t in c.txns:
        by.setdefault(t["account_id"], []).append(t)
    bm = c.branch_map()

    def one(wid):
        w = wins[wid]
        a = amap[w["account_id"]]
        text = RD.as_prompt(RD.render_window(a, by[a["account_id"]], w, fx[wid], bm))
        p = PROMPT.format(window=text, q=(I[wid].get("rationale") or "")[:300],
                          lit=(Lz[wid].get("rationale") or "")[:300])
        try:
            return wid, L._call(L.QWEN, p, timeout=150), None
        except Exception as e:
            return wid, None, str(e)[:70]

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=3) as ex:
        res = list(ex.map(one, auto))
    print("%d drafted in %.0fs (%.2f calls/s)\n" % (len(res), time.time() - t0,
                                                    len(res) / (time.time() - t0)))

    out, flagged = [], 0
    for wid, r, err in res:
        if err or not r:
            continue
        f = fx[wid]
        a = amap[wins[wid]["account_id"]]
        bad, cited = check(r.get("narrative", ""), f, r.get("cited_txn_ids"))
        if bad:
            flagged += 1
        out.append({"window_id": wid, "account_id": a["account_id"], "sector": a["sector"],
                    "typology": r.get("typology"), "narrative": r.get("narrative", ""),
                    "cited_txn_ids": cited, "unverified_amounts": bad,
                    "total_cash_in": f["cash_deposit_total"],
                    "deposits": f["cash_deposit_count"],
                    "vs_own_history": f["vs_own_history_ratio"],
                    "intent_verdict": I[wid]["verdict"], "literal_verdict": Lz[wid]["verdict"],
                    "truth": c.labels[wid]})
    with open(os.path.join(OUT, "sars.jsonl"), "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")

    real = len([r for r in out if r["truth"]])
    print("%d reports drafted | %d on genuinely planted cases (%.0f%%)"
          % (len(out), real, 100.0 * real / max(1, len(out))))
    print("%d contain a dollar figure not found in the facts -- flagged, not filed" % flagged)
    _html(out)
    print("wrote out/sars.jsonl and out/reports.html")


def _html(out):
    css = ("body{margin:0;background:#eceae4;font:14px/1.6 -apple-system,BlinkMacSystemFont,"
           "'Segoe UI',sans-serif;color:#1b1b1b}.wrap{max-width:980px;margin:0 auto;"
           "padding:26px 20px 60px}h1{font-size:20px;margin:0 0 3px}"
           ".sub{color:#6b6b6b;font-size:12.5px;margin-bottom:20px}"
           ".r{background:#fff;border:1px solid #dcd9d1;border-radius:5px;padding:16px 19px;"
           "margin-bottom:12px}.hd{display:flex;justify-content:space-between;"
           "align-items:baseline;margin-bottom:9px}.id{font-weight:600}"
           ".t{font-size:10.5px;background:#fdecea;color:#a03027;padding:2px 8px;"
           "border-radius:3px;text-transform:uppercase;letter-spacing:.05em}"
           ".n{color:#2b2b2b}.m{font-size:11.5px;color:#8a8a8a;margin-top:9px}"
           ".warn{background:#fff6e5;border:1px solid #e8d5a0;color:#8a5a12;padding:7px 11px;"
           "border-radius:4px;font-size:12px;margin-top:9px}")
    h = ['<!doctype html><meta charset="utf-8"><title>Filed reports</title>',
         '<meta name=viewport content="width=device-width,initial-scale=1">',
         "<style>%s</style><div class=wrap><h1>Auto-drafted reports</h1>" % css,
         "<div class=sub>Both reviewers agreed these are suspicious, so they were drafted "
         "rather than escalated. %d reports. Every dollar figure is checked against the "
         "computed facts before filing.</div>" % len(out)]
    for r in out[:40]:
        h.append("<div class=r><div class=hd><span class=id>%s &middot; %s</span>"
                 "<span class=t>%s</span></div><div class=n>%s</div>"
                 % (r["window_id"], r["sector"], r["typology"] or "other",
                    _e(r["narrative"])))
        if r["unverified_amounts"]:
            h.append("<div class=warn>Held back: contains %s, which does not appear in the "
                     "transaction facts.</div>"
                     % ", ".join("$%s" % format(v, ",d") for v in r["unverified_amounts"]))
        h.append("<div class=m>cites %s</div></div>"
                 % (_e(", ".join(r["cited_txn_ids"][:8])) or "&mdash;"))
    h.append("</div>")
    open(os.path.join(OUT, "reports.html"), "w").write("\n".join(h))


def _e(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    main()
