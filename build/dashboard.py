"""One console. Queue on the left, case on the right -- the layout an AML analyst
actually works in, not a scrolling document.

    python3 build/dashboard.py     -> out/console.html

Two case types share the queue because they are two outcomes of one pipeline:
  ESCALATION  the two models disagreed -> a human rules on it
  FILED       both agreed suspicious   -> drafted as a report

The report view is modelled on FinCEN Form 111: Part I Subject Information, Part II
Suspicious Activity Information, Part V Narrative. Real SARs are a form with a prose
narrative inside it, not an essay, and the form is what makes the amounts checkable.
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpus as C, facts as F, render as RD, rules as R

# every stage rebuilds the corpus; single-threaded that was 34s each time on a
# 20-core box. Per-account RNG makes the parallel build byte-identical.
import multiprocessing
WORKERS = max(1, multiprocessing.cpu_count() - 2)


OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
TYPOLOGY_LABEL = {
    "structuring": "Structuring / evasion of reporting threshold",
    "funnel": "Funnel account activity",
    "layering": "Layering",
    "pass_through": "Pass-through / mule account",
    "cuckoo_smurfing": "Third-party deposits (cuckoo smurfing)",
    "velocity_spike": "Sudden change in account activity",
    "other": "Other suspicious activity",
}


def money(x):
    return "$%s" % format(int(round(x)), ",d")


def trim(s, n=340):
    s = (s or "").strip()
    if len(s) > n:
        s = s[:n]
    i = max(s.rfind(". "), s.rfind("! "), s.rfind("? "))
    return s[:i + 1] if i > 60 else s.rstrip(" ,;:-") + ("…" if s else "")


def bullets(s, limit=3):
    t, parts, cur = trim(s), [], ""
    for ch in t:
        cur += ch
        if ch in ".!?" and len(cur.strip()) > 25:
            parts.append(cur.strip())
            cur = ""
    if cur.strip():
        parts.append(cur.strip())
    return parts[:limit] or [t]


def rows_for(w, a, fx, by, bm, limit=14):
    inw = [t for t in by[a["account_id"]] if w["start"] <= t["day"] <= w["end"]]
    inw.sort(key=lambda t: (t["day"], t["hour"]))
    out = []
    for t in inw[:limit]:
        out.append({"id": t["txn_id"], "day": t["day"] - w["start"] + 1,
                    "dir": t["direction"], "ch": t["channel"],
                    "amt": round(t["amount"], 2), "br": t["branch"],
                    "loc": "%s, %s" % (bm[t["branch"]]["county"], bm[t["branch"]]["state"]),
                    "cp": t.get("counterparty") or ""})
    return out, len(inw)


def main():
    rows = [json.loads(l) for l in open(os.path.join(OUT, "panel.jsonl"))]
    I = dict((r["window_id"], r) for r in rows if r["stage"] == "intent" and r["verdict"])
    Lz = dict((r["window_id"], r) for r in rows if r["stage"] == "literal" and r["verdict"])
    sars = {}
    p = os.path.join(OUT, "sars.jsonl")
    if os.path.exists(p):
        for line in open(p):
            r = json.loads(line)
            sars[r["window_id"]] = r

    rule = json.load(open(os.path.join(OUT, "rule_best.json")))
    c = C.Corpus(seed=7, n_accounts=1200, base_rate=0.01, workers=WORKERS).build()
    wins = dict((w["window_id"], w) for w in c.windows())
    fx = F.all_facts(c.accounts, c.txns, list(wins.values()))
    amap = dict((a["account_id"], a) for a in c.accounts)
    by = {}
    for t in c.txns:
        by.setdefault(t["account_id"], []).append(t)
    bm = c.branch_map()

    cases = []
    for wid, lit in Lz.items():
        q = I.get(wid)
        if not q:
            continue
        w, a, f = wins[wid], amap[wins[wid]["account_id"]], fx[wid]
        agree = q["verdict"] == lit["verdict"]
        if agree and q["verdict"] != "suspicious":
            continue
        kind = ("filed" if agree else
                ("split" if q["verdict"] == "suspicious" else "audit"))
        txns, ntot = rows_for(w, a, fx, by, bm)
        fired = R.fires(rule, f)[1]
        cases.append({
            "id": wid, "kind": kind, "acct": a["account_id"], "sector": a["sector"],
            "opened": a["opened_days_ago"], "branches": a["branches"],
            "cash": f["cash_deposit_total"], "n": f["cash_deposit_count"],
            "lo": f["min_deposit_amount"], "hi": f["max_deposit_amount"],
            "out": f["outbound_total"], "outn": f["outbound_count"],
            "kept": int(round(100 * f["end_balance_ratio"])),
            "vs": f["vs_own_history_ratio"], "nb": f["new_branch_count"],
            "br": f["distinct_branches"], "st": f["distinct_states"],
            "span": f["span_days"],
            "rule": R.predicate_gloss(rule, fired[0]) if fired else "",
            "q": {"v": q["verdict"], "p": trim(q.get("pattern", ""), 200),
                  "b": bullets(q.get("rationale", "")), "c": (q.get("cited") or [])[:6]},
            "l": {"v": lit["verdict"], "p": trim(lit.get("pattern", ""), 200),
                  "b": bullets(lit.get("rationale", "")), "c": (lit.get("cited") or [])[:6]},
            "sar": sars.get(wid), "txns": txns, "ntot": ntot,
        })
    cases.sort(key=lambda x: ({"filed": 0, "audit": 1, "split": 2}[x["kind"]], x["id"]))

    n_filed = len([x for x in cases if x["kind"] == "filed"])
    n_audit = len([x for x in cases if x["kind"] == "audit"])
    n_split = len([x for x in cases if x["kind"] == "split"])
    html = TEMPLATE.replace("__DATA__", json.dumps(cases, separators=(",", ":"))) \
                   .replace("__FILED__", str(n_filed)).replace("__AUDIT__", str(n_audit)) \
                   .replace("__SPLIT__", str(n_split)) \
                   .replace("__REVIEWED__", str(len(I)))
    open(os.path.join(OUT, "console.html"), "w").write(html)
    print("console.html: %d cases (%d filed, %d audit-caught, %d splits)"
          % (len(cases), n_filed, n_audit, n_split))


TEMPLATE = r"""<!doctype html><meta charset="utf-8"><title>AML review console</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
*{box-sizing:border-box}
body{margin:0;background:#f2f1ed;color:#16181d;font:13.5px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif}
.top{background:#1d2330;color:#e8eaef;padding:11px 20px;display:flex;align-items:center;gap:26px;flex-wrap:wrap}
.top h1{font-size:14.5px;margin:0;font-weight:600;letter-spacing:.01em}
.top .m{display:flex;gap:22px;margin-left:auto;flex-wrap:wrap}
.top .m div{font-size:11px;color:#98a0b0}
.top .m b{display:block;font-size:16px;color:#fff;font-weight:600}
.top .m b.g{color:#5fd08a}
.shell{display:grid;grid-template-columns:376px 1fr;height:calc(100vh - 47px)}
@media(max-width:900px){.shell{grid-template-columns:1fr;height:auto}}
.list{background:#fff;border-right:1px solid #d9d7d1;overflow-y:auto}
.tabs{display:flex;border-bottom:1px solid #e6e4de;position:sticky;top:0;background:#fff;z-index:2}
.tabs button{flex:1;border:0;background:none;padding:10px 6px;font:inherit;font-size:12px;
 color:#6b7280;cursor:pointer;border-bottom:2px solid transparent}
.tabs button.on{color:#16181d;border-bottom-color:#c0392b;font-weight:600}
.row{padding:10px 15px;border-bottom:1px solid #f0eeea;cursor:pointer;display:flex;gap:10px}
.row:hover{background:#faf9f7}.row.on{background:#fdf2f0;box-shadow:inset 3px 0 0 #c0392b}
.row .bar{width:3px;border-radius:2px;flex:0 0 3px}
.bar.filed{background:#c0392b}.bar.audit{background:#c9821b}.bar.split{background:#7c8797}
.row .t{flex:1;min-width:0}
.row .id{font:12px ui-monospace,Menlo,monospace;font-weight:600}
.row .s{color:#6b7280;font-size:11.5px;margin-top:1px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.row .a{text-align:right;font-size:12px;font-weight:600}
.row .k{font-size:10px;color:#8b93a1;text-align:right;margin-top:2px}
.pane{overflow-y:auto;padding:22px 26px 60px}
.empty{color:#8b93a1;padding:60px 0;text-align:center}
.hd{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;margin-bottom:3px}
.hd h2{margin:0;font-size:18px;font:600 18px ui-monospace,Menlo,monospace}
.tag{font-size:10.5px;padding:2px 8px;border-radius:3px;letter-spacing:.05em;text-transform:uppercase}
.tag.filed{background:#fbe9e7;color:#a03027}.tag.audit{background:#fdf3e0;color:#8a5a12}
.tag.split{background:#eef0f3;color:#54606f}
.sub{color:#6b7280;font-size:12.5px;margin-bottom:18px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:1100px){.two{grid-template-columns:1fr}}
.box{background:#fff;border:1px solid #ddd9d2;border-radius:5px;padding:13px 15px}
.box .lab{font-size:10.5px;color:#7b8391;text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px}
.v{font-weight:700;font-size:13px}.sus{color:#a03027}.ben{color:#1a6b3c}
.pat{color:#4b5563;margin:5px 0 0;font-size:12.5px}
ul{margin:7px 0 0;padding-left:16px}li{margin-bottom:3px;color:#3f4653;font-size:12.5px}
.cites{font:11px ui-monospace,Menlo,monospace;color:#8b93a1;margin-top:7px}
/* --- the report, modelled on FinCEN Form 111 --- */
.sar{background:#fff;border:1px solid #c9c4ba;border-radius:4px;margin-top:16px}
.sarhd{background:#2b3242;color:#fff;padding:11px 16px;border-radius:3px 3px 0 0;
 display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px}
.sarhd b{font-size:13px;letter-spacing:.02em}
.sarhd span{font:11px ui-monospace,Menlo,monospace;color:#a9b2c2}
.part{border-top:1px solid #e6e2da}
.part h4{margin:0;padding:7px 16px;background:#f4f2ee;font-size:10.5px;color:#5b6472;
 text-transform:uppercase;letter-spacing:.06em;font-weight:700}
.fields{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:0;padding:0}
.f{padding:9px 16px;border-right:1px solid #f0ece4;border-bottom:1px solid #f0ece4}
.f .k{font-size:10px;color:#8b93a1;text-transform:uppercase;letter-spacing:.04em}
.f .val{font-size:13px;margin-top:2px}
.f .val.mono{font-family:ui-monospace,Menlo,monospace;font-size:12px}
.chk{padding:10px 16px;display:flex;flex-wrap:wrap;gap:7px}
.chk span{font-size:11.5px;padding:3px 9px;border:1px solid #ddd9d2;border-radius:3px;color:#8b93a1}
.chk span.on{background:#fbe9e7;border-color:#e8b4ae;color:#a03027;font-weight:600}
.nar{padding:14px 18px;font:13.5px/1.72 Georgia,'Times New Roman',serif;color:#1b1f27}
.ver{margin:0 16px 14px;padding:7px 11px;background:#eef7f0;border:1px solid #cfe6d6;
 border-radius:3px;font-size:11.5px;color:#1a6b3c}
.ver.bad{background:#fff6e5;border-color:#e8d5a0;color:#8a5a12}
table{width:100%;border-collapse:collapse;font-size:12px}
th{background:#f7f5f1;text-align:left;padding:6px 10px;font-size:10px;color:#7b8391;
 text-transform:uppercase;letter-spacing:.04em;border-bottom:1px solid #e6e2da}
td{padding:5px 10px;border-bottom:1px solid #f2efe9}
td.mono{font-family:ui-monospace,Menlo,monospace;font-size:11.5px}
td.num{text-align:right;font-variant-numeric:tabular-nums}
tr.cited td{background:#fffaf0}
.acts{margin-top:16px;display:flex;gap:8px;flex-wrap:wrap}
.btn{padding:7px 15px;border:1px solid #d2cec6;border-radius:4px;background:#fff;
 font:inherit;font-size:12.5px;color:#3f4653;cursor:pointer}
.btn.p{background:#a03027;border-color:#a03027;color:#fff}
</style>
<div class=top>
  <h1>AML Review Console</h1>
  <div class=m>
    <div>reviewed<b>__REVIEWED__</b></div>
    <div>rule alone<b>0.060</b></div>
    <div>both models agree<b class=g>0.338</b></div>
    <div>filed<b>__FILED__</b></div>
    <div>escalated<b>__SPLIT__ + __AUDIT__</b></div>
  </div>
</div>
<div class=shell>
  <div class=list>
    <div class=tabs>
      <button data-f=all class=on>All</button>
      <button data-f=filed>Filed (__FILED__)</button>
      <button data-f=audit>Audit (__AUDIT__)</button>
      <button data-f=split>Split (__SPLIT__)</button>
    </div>
    <div id=rows></div>
  </div>
  <div class=pane id=pane><div class=empty>Select a case from the queue.</div></div>
</div>
<script>
const CASES = __DATA__;
const $ = s => document.querySelector(s);
const m = v => '$' + Math.round(v).toLocaleString();
const esc = s => String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
// same words as the live monitor: Filed / Panel split / Audit caught
const KIND = {filed:'Filed', audit:'Audit caught', split:'Panel split'};
let filter = 'all', sel = null;

function list(){
  const f = CASES.filter(c => filter==='all' || c.kind===filter);
  $('#rows').innerHTML = f.map(c => `
    <div class="row ${c.id===sel?'on':''}" data-id="${c.id}">
      <div class="bar ${c.kind}"></div>
      <div class=t><div class=id>${c.id}</div><div class=s>${esc(c.sector)}</div></div>
      <div><div class=a>${m(c.cash)}</div><div class=k>${KIND[c.kind]}</div></div>
    </div>`).join('') || '<div class=empty>None.</div>';
  document.querySelectorAll('.row').forEach(r =>
    r.onclick = () => { sel = r.dataset.id; list(); show(CASES.find(c=>c.id===sel)); });
}

function lensBox(title, model, d){
  return `<div class=box><div class=lab>${title} &middot; ${model}</div>
    <div class="v ${d.v==='suspicious'?'sus':'ben'}">${d.v.toUpperCase()}</div>
    ${d.p?`<div class=pat>${esc(d.p)}</div>`:''}
    <ul>${d.b.map(b=>`<li>${esc(b)}</li>`).join('')}</ul>
    ${d.c.length?`<div class=cites>cites ${esc(d.c.join(', '))}</div>`:''}</div>`;
}

function txnTable(c){
  const cited = new Set([...(c.q.c||[]), ...(c.l.c||[])]);
  return `<table><tr><th>Day</th><th>Transaction</th><th>Type</th><th class=num>Amount</th>
    <th>Branch</th><th>Counterparty</th></tr>
    ${c.txns.map(t=>`<tr class="${cited.has(t.id)?'cited':''}">
      <td class=num>${t.day}</td><td class=mono>${t.id}</td>
      <td>${t.ch} ${t.dir}</td><td class="num">${m(t.amt)}</td>
      <td>${t.br} <span style="color:#9aa1ad">${esc(t.loc)}</span></td>
      <td class=mono>${esc(t.cp)}</td></tr>`).join('')}
    ${c.ntot>c.txns.length?`<tr><td colspan=6 style="color:#9aa1ad">
      ${c.ntot-c.txns.length} further transactions in this period</td></tr>`:''}</table>`;
}

const TYPES = {structuring:'Structuring', funnel:'Funnel account', layering:'Layering',
  pass_through:'Pass-through / mule', cuckoo_smurfing:'Third-party deposits',
  velocity_spike:'Change in activity', other:'Other'};

function sarForm(c){
  const s = c.sar;
  const bad = (s.unverified_amounts||[]).length;
  return `<div class=sar>
    <div class=sarhd><b>FinCEN Form 111 &mdash; Suspicious Activity Report</b>
      <span>DRAFT &middot; ${c.id} &middot; prepared automatically</span></div>

    <div class=part><h4>Part I &mdash; Subject Information</h4><div class=fields>
      <div class=f><div class=k>Subject type</div><div class=val>Entity &mdash; account holder</div></div>
      <div class=f><div class=k>Account number</div><div class="val mono">${c.acct}</div></div>
      <div class=f><div class=k>Business type</div><div class=val>${esc(c.sector)}</div></div>
      <div class=f><div class=k>Relationship</div><div class=val>Depositor, ${c.opened} days</div></div>
      <div class=f><div class=k>Branches of record</div><div class="val mono">${c.branches.join(', ')}</div></div>
      <div class=f><div class=k>Role in activity</div><div class=val>Subject</div></div>
    </div></div>

    <div class=part><h4>Part II &mdash; Suspicious Activity Information</h4><div class=fields>
      <div class=f><div class=k>Amount involved</div><div class=val><b>${m(c.cash)}</b></div></div>
      <div class=f><div class=k>Period of activity</div><div class=val>${c.span} days</div></div>
      <div class=f><div class=k>Cash transactions</div><div class=val>${c.n} deposits, ${m(c.lo)}&ndash;${m(c.hi)}</div></div>
      <div class=f><div class=k>Funds disposition</div><div class=val>${c.outn} transfer(s), ${m(c.out)} out; ${c.kept}% retained</div></div>
      <div class=f><div class=k>Locations</div><div class=val>${c.br} branch(es), ${c.st} state(s)${c.nb?`, ${c.nb} not previously used`:''}</div></div>
      <div class=f><div class=k>Against account history</div><div class=val>${c.vs.toFixed(2)}&times; normal volume</div></div>
    </div>
    <div class=chk>${Object.entries(TYPES).map(([k,v])=>
      `<span class="${k===s.typology?'on':''}">${v}</span>`).join('')}</div></div>

    <div class=part><h4>Part V &mdash; Suspicious Activity Narrative</h4>
      <div class=nar>${esc(s.narrative)}</div>
      <div class="ver ${bad?'bad':''}">${bad
        ? `Held from filing: contains ${s.unverified_amounts.map(m).join(', ')}, not present in the transaction record.`
        : 'Every monetary figure in this narrative was matched against the transaction record before drafting.'}</div>
    </div>

    <div class=part><h4>Supporting transactions</h4>${txnTable(c)}</div>
  </div>`;
}

function show(c){
  const why = c.kind==='filed' ? 'Both reviewers concluded suspicious &mdash; drafted for filing'
    : c.kind==='audit' ? 'The first reviewer cleared this; the second flagged it'
    : 'The two reviewers disagree';
  $('#pane').innerHTML = `
    <div class=hd><h2>${c.id}</h2><span class="tag ${c.kind}">${KIND[c.kind]}</span></div>
    <div class=sub>${esc(c.sector)} &middot; account ${c.acct} &middot; opened ${c.opened} days ago
      &mdash; ${why}</div>
    <div class=two>
      ${lensBox('Intent &mdash; legitimate purpose?','Qwen3.6-35B',c.q)}
      ${lensBox("Literal &mdash; does the rule's evidence hold?",'Nemotron-3-Nano',c.l)}
    </div>
    ${c.rule?`<div class=box style="margin-top:14px"><div class=lab>Rule that fired</div>
      <div style="font-size:12.5px;color:#3f4653">${esc(c.rule)}</div></div>`:''}
    ${c.sar ? sarForm(c) : `<div class=box style="margin-top:16px">
      <div class=lab>Supporting transactions</div>${txnTable(c)}</div>`}
    <div class=acts>${c.kind==='filed'
      ? '<button class="btn p">Submit report</button><button class=btn>Return for review</button>'
      : '<button class="btn p">File report</button><button class=btn>Close &mdash; no action</button><button class=btn>Request more information</button>'}</div>`;
  $('#pane').scrollTop = 0;
}

document.querySelectorAll('.tabs button').forEach(b => b.onclick = () => {
  document.querySelectorAll('.tabs button').forEach(x=>x.classList.remove('on'));
  b.classList.add('on'); filter = b.dataset.f; list();
});
list();
if (CASES.length) { sel = CASES[0].id; list(); show(CASES[0]); }
</script>
"""

if __name__ == "__main__":
    main()
