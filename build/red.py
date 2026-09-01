"""RED -- constructs a transaction sequence that launders money WITHOUT tripping the
current rule. The adversary in the loop, and the strongest fifteen seconds of the pitch.

Red never needs a model call to know whether it succeeded. fires() answers that in
Python, instantly, deterministically -- which is the whole reason RED_BLUE_SPEC section 2
made L0 a predicate rule instead of a prompt. is_valid_evasion() is the referee.

    python3 build/red.py --tries 8

Red is also the OVERFITTING DEFENCE. If blue's rule had memorised the generator's
fingerprint rather than learning laundering, red's mutations would walk straight
through it. Red succeeding cheaply is bad news about the rule, and worth knowing.
"""
import json, os, sys, urllib.request
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import facts as F, rules as R

ENDPOINT = ("http://localhost:8000/v1/chat/completions", "qwen")
MAX_TXNS = 18

TXN = {"type": "object", "properties": {
    "day": {"type": "integer"},
    "hour": {"type": "integer"},
    "amount": {"type": "number"},
    "direction": {"type": "string", "enum": ["in", "out"]},
    "channel": {"type": "string", "enum": ["cash", "check", "ach", "wire"]},
    "branch": {"type": "string", "maxLength": 8},
    "counterparty": {"type": "string", "maxLength": 12}},
    "required": ["day", "hour", "amount", "direction", "channel", "branch", "counterparty"]}

TOOLS = [{"type": "function", "function": {
    "name": "submit_sequence",
    "description": "Submit a transaction sequence that evades the rule.",
    "parameters": {"type": "object", "properties": {
        "transactions": {"type": "array", "maxItems": MAX_TXNS, "items": TXN},
        "weakness": {"type": "string", "maxLength": 300}},
        "required": ["transactions", "weakness"]}}}]

PROMPT = """You are testing a bank's transaction monitoring system for gaps, so the bank
can close them. This is a controlled exercise against synthetic data.

THE RULE CURRENTLY IN FORCE
{gloss}

It is evaluated in code against facts computed from the account's transactions. It fires
if ANY predicate matches. These are the exact predicates:
{rule_json}

FACTS THE SYSTEM CAN SEE -- it is blind to everything else:
{fact_list}

THE ACCOUNT YOU CONTROL
  {sector} business, cash-intensity {ci}, opened {age} days ago
  Registered branches: {branches}
  Other branches that exist in this bank: {other}
  Over the previous periods it averaged {avg_dep} cash deposits per fortnight
  totalling about ${avg_total}, and its usual counterparties are {cps}

OBJECTIVE
Move at least ${target} of cash into this account and back out to a destination you
control, inside a {days}-day window, WITHOUT the rule above firing.

CONSTRAINTS -- a sequence violating any of these is discarded by an automatic referee
1. Total cash in must be at least ${target}.
2. Essentially all of it must leave: end balance within 2% of what came in.
3. At most {maxtxn} transactions. Days are numbered 1 to {days}.
4. It must look like something a {sector} business could plausibly do. No 40 cash
   deposits a week from a dental practice.
5. Use only branches listed above.

The rule watches specific facts at specific thresholds. Find the one it depends on most,
and keep your sequence on the safe side of it while still moving the money.

Return the sequence with the tool, and state in one sentence which fact you defeated
and how."""


def cost(seq):
    """Real launderers face real costs: mules burn, branch visits take time, long
    spans tie up capital. Without a budget red's optimal move is 10,000 deposits of
    $8 across 400 accounts -- which evades everything and is not laundering."""
    branches = len(set(t["branch"] for t in seq))
    deps = len([t for t in seq if t["direction"] == "in"])
    span = (max(t["day"] for t in seq) - min(t["day"] for t in seq) + 1) if seq else 0
    return 0.5 * branches + 0.1 * deps + 0.05 * span


def to_txns(account, seq, window_start, branch_map):
    out = []
    for i, t in enumerate(seq):
        b = t["branch"] if t["branch"] in branch_map else account["branches"][0]
        meta = branch_map[b]
        out.append({"txn_id": "RED%04d" % (i + 1), "account_id": account["account_id"],
                    "day": window_start + max(0, int(t["day"]) - 1),
                    "hour": max(0, min(23, int(t["hour"]))),
                    "amount": round(float(t["amount"]), 2),
                    "direction": t["direction"], "channel": t["channel"], "branch": b,
                    "state": meta["state"], "fips": meta["fips"],
                    "counterparty": t.get("counterparty"), "third_party": False})
    return sorted(out, key=lambda t: (t["day"], t["hour"]))


def repair(seq_txns, target):
    """Close the arithmetic, keep the strategy.

    Measured failure modes across two runs: undershot the target by $600, left the money
    sitting in the account, moved 1.6x what came in. None of those is a failure to find an
    evasion -- they are a model that designs a good pattern and cannot sum twelve numbers.

    So: scale the deposits to the target if it undershot, and set the terminal outbound to
    exactly what came in. Real launderers move what is there; they do not do mental
    arithmetic either. The referee still has to pass afterwards -- scaling can push a
    deposit over a threshold and make the rule fire, and then the attempt fails honestly.

    Every repair is recorded and reported. Nothing here is silent.
    """
    notes = []
    ins = [t for t in seq_txns if t["direction"] == "in"]
    outs = [t for t in seq_txns if t["direction"] == "out"]
    if not ins:
        return seq_txns, ["no inbound to repair"]
    cash_in = sum(t["amount"] for t in ins if t["channel"] == "cash")
    if 0 < cash_in < target:
        k = target / cash_in
        for t in ins:
            t["amount"] = round(t["amount"] * k, 2)
        notes.append("scaled deposits x%.3f to reach the $%s target" % (k, format(target, ",d")))
    inflow = sum(t["amount"] for t in ins)
    if not outs:
        last = max(ins, key=lambda t: (t["day"], t["hour"]))
        seq_txns = seq_txns + [dict(last, txn_id="REDOUT", direction="out", channel="wire",
                                    amount=round(inflow, 2), hour=min(23, last["hour"] + 2),
                                    counterparty="DEST")]
        notes.append("added the terminal transfer red omitted")
    else:
        term = max(outs, key=lambda t: (t["day"], t["hour"]))
        other = sum(t["amount"] for t in outs if t is not term)
        want = round(inflow - other, 2)
        if want <= 0:
            # red moved out more than came in. A negative "repair" would balance the
            # books arithmetically and be nonsense -- refuse it and let the attempt
            # fail on money_left, which is the honest outcome.
            return sorted(seq_txns, key=lambda t: (t["day"], t["hour"])), \
                   ["UNREPAIRABLE: outbound exceeds inbound, refused to write a negative transfer"]
        if abs(term["amount"] - want) > 1.0:
            notes.append("closed the balance: terminal transfer $%s -> $%s"
                         % (format(int(term["amount"]), ",d"), format(int(want), ",d")))
            term["amount"] = want
    return sorted(seq_txns, key=lambda t: (t["day"], t["hour"])), notes


def is_valid_evasion(rule, account, prior_txns, seq_txns, window, target, budget=6.0):
    """The referee. All five, in Python, no model call. -> (ok, detail)"""
    d = {}
    cash_in = sum(t["amount"] for t in seq_txns
                  if t["direction"] == "in" and t["channel"] == "cash")
    inflow = sum(t["amount"] for t in seq_txns if t["direction"] == "in")
    outflow = sum(t["amount"] for t in seq_txns if t["direction"] == "out")
    d["moved"] = round(cash_in, 2)
    d["moves_enough"] = cash_in >= target
    d["end_balance"] = round(inflow - outflow, 2)
    d["money_left"] = abs(inflow - outflow) / max(1.0, target) < 0.02
    d["cost"] = round(cost(seq_txns), 2)
    d["within_budget"] = d["cost"] <= budget
    per_day = {}
    for t in seq_txns:
        if t["direction"] == "in":
            per_day[t["day"]] = per_day.get(t["day"], 0) + 1
    d["max_deposits_per_day"] = max(per_day.values()) if per_day else 0
    d["plausible"] = d["max_deposits_per_day"] <= 4

    merged = sorted(prior_txns + seq_txns, key=lambda t: (t["day"], t["hour"]))
    fx = F.window_facts(account, merged, window, F.prior_state(merged, window))
    fired, which = R.fires(rule, fx)
    d["rule_fired"] = fired
    d["fired_predicates"] = which
    d["evades"] = not fired
    d["facts"] = fx
    ok = (d["moves_enough"] and d["money_left"] and d["within_budget"]
          and d["plausible"] and d["evades"])
    return ok, d


def why_it_evades(rule, fx):
    """Explain the evasion from the ARITHMETIC, not from red's own account of it.

    Red's prose is unreliable -- it thinks out loud and contradicts itself mid-sentence.
    But the facts are computed and the thresholds are in the rule, so the margin on every
    clause is derivable. This is what goes on stage: not "the model says it evaded" but
    "here is the number it kept on the safe side, and by how much"."""
    out = []
    for p in rule["predicates"]:
        blocked = []
        for c in p["all"]:
            v = fx.get(c["fact"])
            if v is None or not R.OPS[c["op"]](v, c["value"]):
                blocked.append((c["fact"], v, c["op"], c["value"]))
        out.append({"predicate": p["id"], "fired": not blocked,
                    "blocked_by": blocked[:3],
                    "clauses_total": len(p["all"])})
    return out


def evasion_lines(rule, fx):
    lines = []
    for e in why_it_evades(rule, fx):
        if e["fired"]:
            lines.append("  %-4s FIRED" % e["predicate"])
            continue
        f, v, op, thr = e["blocked_by"][0]
        lines.append("  %-4s blocked: %s is %s, needs %s %s"
                     % (e["predicate"], f, _fmt(v), op, _fmt(thr)))
    return lines


def _fmt(v):
    if v is None:
        return "n/a"
    if abs(v) >= 1000:
        return "$%s" % format(int(v), ",d")
    return ("%.2f" % v).rstrip("0").rstrip(".")


def attack(rule, account, hist, branch_map, target, days=14, temperature=0.9, timeout=180):
    others = [b for b in sorted(branch_map) if b not in account["branches"]][:6]
    prompt = PROMPT.format(
        gloss=R.gloss_from_json(rule),
        rule_json=json.dumps([{"id": p["id"], "clauses": p["all"]} for p in rule["predicates"]],
                             indent=1),
        fact_list=", ".join(F.FACT_NAMES),
        sector=account["sector"], ci=account["cash_intensity"],
        age=account["opened_days_ago"], branches=", ".join(account["branches"]),
        other=", ".join(others), avg_dep=hist["avg_dep"], avg_total=hist["avg_total"],
        cps=", ".join(hist["cps"][:3]) or "none on record",
        target=target, days=days, maxtxn=MAX_TXNS)
    url, model = ENDPOINT
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "tools": TOOLS,
            "tool_choice": {"type": "function", "function": {"name": "submit_sequence"}},
            "max_tokens": 1600, "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    tc = d["choices"][0]["message"].get("tool_calls") or []
    if not tc:
        raise RuntimeError("no tool_call; finish=%s" % d["choices"][0]["finish_reason"])
    return json.loads(tc[0]["function"]["arguments"])
