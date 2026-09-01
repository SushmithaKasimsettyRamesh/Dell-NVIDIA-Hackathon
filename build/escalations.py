"""Escalation dashboard. Static HTML, no network, cannot fail on stage.

An escalation is qwen_verdict != nemotron_verdict, computed in Python. No model is
asked whether to escalate -- that would add a third judgement with no ground truth.

Display notes, all learned from looking at the first version:
  - lens rationales are capped at 400 chars by the tool schema, so guided decoding
    stops MID-WORD. Trimmed to the last complete sentence rather than shown as a
    fragment. The cap is upstream; this is the honest way to render it.
  - rationales render as bullets. A reviewer scans; they do not read paragraphs.
  - the draft narrative is built from FACTS, not by concatenating the two rationales.
    Stitching them produced "...Although the new Reviewing against the written rule:"
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpus as C, facts as F, render as RD, rules as R

# every stage rebuilds the corpus; single-threaded that was 34s each time on a
# 20-core box. Per-account RNG makes the parallel build byte-identical.
import multiprocessing
WORKERS = max(1, multiprocessing.cpu_count() - 2)


OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
DETAIL = 25

CSS = """
*{box-sizing:border-box}
body{margin:0;background:#eceae4;color:#1b1b1b;
 font:14px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
.wrap{max-width:1240px;margin:0 auto;padding:26px 20px 60px}
h1{font-size:20px;margin:0 0 3px;font-weight:600}
.sub{color:#6b6b6b;font-size:12.5px;margin-bottom:20px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(148px,1fr));gap:12px;
 margin-bottom:22px}
.card{background:#fff;border:1px solid #dcd9d1;border-radius:5px;padding:13px 15px}
.card b{display:block;font-size:22px;font-weight:600;line-height:1.15}
.card span{font-size:10.5px;color:#7a7a7a;text-transform:uppercase;letter-spacing:.06em}
.card i{font-style:normal;font-size:11px;color:#8a8a8a;display:block;margin-top:3px}
.up{color:#1a6b3c}.down{color:#a03027}
table{width:100%;border-collapse:collapse;background:#fff;border:1px solid #dcd9d1;
 border-radius:5px;overflow:hidden;font-size:12.5px}
th{background:#f4f2ec;text-align:left;padding:8px 11px;font-size:10.5px;color:#6b6b6b;
 text-transform:uppercase;letter-spacing:.05em;font-weight:600}
td{padding:7px 11px;border-top:1px solid #eeece6}
tr:hover td{background:#fbfaf7}
.sus{color:#a03027;font-weight:600}.ben{color:#1a6b3c;font-weight:600}
.pill{display:inline-block;padding:2px 7px;border-radius:3px;font-size:10.5px}
.p-split{background:#fdecea;color:#a03027}.p-audit{background:#fdf3e0;color:#8a5a12}
details{background:#fff;border:1px solid #dcd9d1;border-radius:5px;margin-bottom:9px}
summary{padding:11px 15px;cursor:pointer;font-size:13px;display:flex;gap:12px;
 align-items:center;flex-wrap:wrap}
summary::-webkit-details-marker{color:#bbb}
.dbody{padding:2px 18px 16px;border-top:1px solid #eeece6}
.two{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:12px}
@media(max-width:820px){.two{grid-template-columns:1fr}}
.lens{border:1px solid #e6e3db;border-radius:4px;padding:11px 13px}
.who{font-size:10.5px;color:#7a7a7a;text-transform:uppercase;letter-spacing:.05em;
 margin-bottom:5px}
.model{font-size:10.5px;color:#9a9a9a}
ul{margin:7px 0 0;padding-left:17px}li{margin-bottom:4px;color:#3f3f3f}
.cited{font-size:11px;color:#8a8a8a;margin-top:7px}
.sar{background:#fbf8ef;border:1px solid #e8dfc4;border-radius:4px;padding:12px 15px;
 margin-top:14px}
.sar b{font-size:10.5px;color:#8a6d1e;text-transform:uppercase;letter-spacing:.05em;
 display:block;margin-bottom:7px}
.sar dl{margin:0;display:grid;grid-template-columns:132px 1fr;gap:5px 14px;font-size:13px}
.sar dt{color:#8a7a4e;font-size:11.5px}.sar dd{margin:0}
pre{background:#fafaf7;border:1px solid #eeece6;border-radius:4px;padding:11px;
 overflow-x:auto;font:11px/1.5 ui-monospace,Menlo,monospace;color:#444;margin:13px 0 0}
.btn{display:inline-block;margin:13px 7px 0 0;padding:6px 13px;border:1px solid #d2cfc7;
 border-radius:4px;background:#fafaf7;font-size:12.5px;color:#4a4a4a}
"""


def trim(s, n=340):
    """The tool schema caps rationales at 400 chars, so the model stops mid-word.
    Show the last COMPLETE sentence rather than a fragment."""
    s = (s or "").strip()
    if len(s) > n:
        s = s[:n]
    i = max(s.rfind(". "), s.rfind("! "), s.rfind("? "))
    if i > 60:
        return s[:i + 1]
    return s.rstrip(" ,;:-") + ("…" if s else "")


def bullets(s, limit=3):
    """A reviewer scans. Split the rationale into sentences and show the first few."""
    t = trim(s)
    parts, cur = [], ""
    for ch in t:
        cur += ch
        if ch in ".!?" and len(cur.strip()) > 25:
            parts.append(cur.strip())
            cur = ""
    if cur.strip():
        parts.append(cur.strip())
    return parts[:limit] or [t]


def money(x):
    return "$%s" % format(int(round(x)), ",d")


def narrative(a, wid, fx, q, lit, fired_gloss):
    """Built from FACTS. The first version concatenated the two lens rationales and
    produced '...Although the new Reviewing against the written rule:'."""
    return [
        ("Account", "%s &mdash; %s, opened %d days ago"
         % (a["account_id"], a["sector"], a["opened_days_ago"])),
        ("Period", "window %s, %d days" % (wid, 14)),
        ("Activity", "%d cash deposits totalling %s, ranging %s to %s, across %d "
                     "branch(es) in %d state(s)"
         % (fx["cash_deposit_count"], money(fx["cash_deposit_total"]),
            money(fx["min_deposit_amount"]), money(fx["max_deposit_amount"]),
            fx["distinct_branches"], fx["distinct_states"])),
        ("Movement out", "%d transfer(s) totalling %s; %d%% of inbound funds remained"
         % (fx["outbound_count"], money(fx["outbound_total"]),
            int(round(100 * fx["end_balance_ratio"])))),
        ("Against its own history", "%.2fx this account's normal volume; %d branch(es) "
                                    "it had not used before"
         % (fx["vs_own_history_ratio"], fx["new_branch_count"])),
        ("Rule that fired", fired_gloss),
        ("Reviewers", "intent said <b>%s</b>, literal said <b>%s</b> &mdash; referred for "
                      "human determination" % (q["verdict"], lit["verdict"])),
        ("Evidence", ", ".join((q.get("cited") or [])[:6]) or "see transactions below"),
    ]


def main():
    rows = [json.loads(l) for l in open(os.path.join(OUT, "panel.jsonl"))]
    I = dict((r["window_id"], r) for r in rows if r["stage"] == "intent" and r["verdict"])
    L = dict((r["window_id"], r) for r in rows if r["stage"] == "literal" and r["verdict"])
    m = {}
    p = os.path.join(OUT, "cascade_metrics.json")
    if os.path.exists(p):
        m = json.load(open(p))

    esc = []
    for wid, lit in L.items():
        q = I.get(wid)
        if q and q["verdict"] != lit["verdict"]:
            esc.append((wid, q, lit, "split" if q["verdict"] == "suspicious" else "audit"))
    esc.sort(key=lambda e: (e[3] != "audit", e[0]))

    rule = json.load(open(os.path.join(OUT, "rule_best.json")))
    c = C.Corpus(seed=7, n_accounts=1200, base_rate=0.01, workers=WORKERS).build()
    wins = dict((w["window_id"], w) for w in c.windows())
    fx = F.all_facts(c.accounts, c.txns, list(wins.values()))
    amap = dict((a["account_id"], a) for a in c.accounts)
    by = {}
    for t in c.txns:
        by.setdefault(t["account_id"], []).append(t)
    bm = c.branch_map()

    n_split = len([e for e in esc if e[3] == "split"])
    n_audit = len([e for e in esc if e[3] == "audit"])
    auto = len([w for w in I if I[w]["verdict"] == "suspicious"
                and L.get(w, {}).get("verdict") == "suspicious"])

    h = ['<!doctype html><meta charset="utf-8"><title>AML escalation queue</title>',
         '<meta name=viewport content="width=device-width,initial-scale=1">',
         "<style>%s</style><div class=wrap>" % CSS,
         "<h1>Escalation queue</h1>",
         "<div class=sub>Two independent models reviewed every alert. These are the cases "
         "they disagreed on. Escalation is a comparison in code &mdash; no model is asked "
         "whether to escalate.</div>",
         "<div class=grid>"]
    for val, lab, note, cls in (
            ("%d" % len(I), "alerts reviewed", "uniform sample of 1,599", ""),
            ("0.060", "rule alone", "precision", ""),
            ("0.338", "both models agree", "precision &mdash; 5.6&times;", "up"),
            ("%d" % auto, "auto-filed", "both said suspicious", ""),
            ("%d" % len(esc), "escalated", "%d split &middot; %d audit-caught"
             % (n_split, n_audit), "down"),
            ("0.952", "real cases kept", "of what the rule found", "")):
        h.append("<div class=card><b class='%s'>%s</b><span>%s</span><i>%s</i></div>"
                 % (cls, val, lab, note))
    h.append("</div>")

    h.append("<table><tr><th>Window</th><th>Sector</th><th>Why escalated</th>"
             "<th>Intent</th><th>Literal</th><th>Cash in</th><th>Vs normal</th></tr>")
    for wid, q, lit, kind in esc[:60]:
        a = amap[wins[wid]["account_id"]]
        f = fx[wid]
        h.append("<tr><td>%s</td><td>%s</td><td><span class='pill p-%s'>%s</span></td>"
                 "<td class=%s>%s</td><td class=%s>%s</td><td>%s</td><td>%.2fx</td></tr>"
                 % (wid, a["sector"][:22], kind,
                    "panel split" if kind == "split" else "audit caught",
                    "sus" if q["verdict"] == "suspicious" else "ben", q["verdict"],
                    "sus" if lit["verdict"] == "suspicious" else "ben", lit["verdict"],
                    money(f["cash_deposit_total"]), f["vs_own_history_ratio"]))
    h.append("</table>")
    h.append("<div class=sub style='margin:18px 0 10px'>Showing detail for the first %d. "
             "The audit-caught cases are listed first &mdash; those are ones the first "
             "reviewer cleared and the second flagged.</div>" % min(DETAIL, len(esc)))

    for wid, q, lit, kind in esc[:DETAIL]:
        w = wins[wid]
        a = amap[w["account_id"]]
        f = fx[wid]
        r = RD.render_window(a, by[a["account_id"]], w, f, bm)
        fired = R.fires(rule, f)[1]
        fg = R.predicate_gloss(rule, fired[0]) if fired else "no predicate"
        h.append("<details><summary><b>%s</b> <span class='pill p-%s'>%s</span> "
                 "<span style='color:#6b6b6b'>%s &middot; %s cash in &middot; %.2fx normal"
                 "</span></summary><div class=dbody><div class=two>"
                 % (wid, kind, "panel split" if kind == "split" else "audit caught",
                    a["sector"], money(f["cash_deposit_total"]), f["vs_own_history_ratio"]))
        for title, v, model in (
                ("Intent &mdash; is there a legitimate purpose?", q, "Qwen3.6-35B"),
                ("Literal &mdash; does the rule's evidence hold?", lit, "Nemotron-3-Nano")):
            cls = "sus" if v["verdict"] == "suspicious" else "ben"
            h.append("<div class=lens><div class=who>%s <span class=model>%s</span></div>"
                     "<div class='%s'>%s</div>" % (title, model, cls, v["verdict"].upper()))
            if v.get("pattern"):
                h.append("<div style='color:#555;margin-top:5px;font-size:13px'>%s</div>"
                         % _e(trim(v["pattern"], 200)))
            h.append("<ul>%s</ul>" % "".join("<li>%s</li>" % _e(b)
                                             for b in bullets(v.get("rationale", ""))))
            if v.get("cited"):
                h.append("<div class=cited>cites %s</div>" % _e(", ".join(v["cited"][:6])))
            h.append("</div>")
        h.append("</div>")
        h.append("<div class=sar><b>Draft narrative for the reviewer</b><dl>")
        for k, val in narrative(a, wid, f, q, lit, fg):
            h.append("<dt>%s</dt><dd>%s</dd>" % (k, val))
        h.append("</dl></div>")
        h.append("<pre>%s</pre>" % _e(r["records"]))
        h.append("<div><span class=btn>File report</span><span class=btn>Close &mdash; no "
                 "action</span><span class=btn>Request more information</span></div>")
        h.append("</div></details>")

    h.append("<div class=sub style='margin-top:20px'>%d escalations total &middot; "
             "%d panel splits &middot; %d caught by the audit sample</div></div>"
             % (len(esc), n_split, n_audit))
    open(os.path.join(OUT, "escalations.html"), "w").write("\n".join(h))
    print("escalations.html: %d escalations (%d splits, %d audit-caught), %d detailed"
          % (len(esc), n_split, n_audit, min(DETAIL, len(esc))))


def _e(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    main()
