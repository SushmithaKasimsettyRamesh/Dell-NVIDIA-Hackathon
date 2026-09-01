"""BLUE -- proposes an improved L0 rule. The model's ONLY job in the loop.

Scoring a rule is fires() in Python, free, over every window. So blue is never asked
to classify anything; it is asked to write a better rule. That is why 100 iterations
cost 400 calls and not 3,000.

Blue reads FACTS AND DISTRIBUTIONS, never narrative. It is doing threshold search over
a 36-dimensional vocabulary, and a prose summary would be strictly worse input.

Every candidate goes through validate() before it is allowed near the scorer. A rewrite
that names a fact that does not exist, or an operator that does not exist, or nests
itself into a shape fires() would raise on, is DISCARDED and logged -- never crashed on,
never silently coerced into something that scores.
"""
import json, statistics, urllib.request
from typing import Dict, List, Optional

import facts as F
import rules as R

ENDPOINT = ("http://localhost:8000/v1/chat/completions", "qwen")
MAX_PREDICATES = 4
MAX_CLAUSES = 5

CLAUSE = {"type": "object", "properties": {
    "fact": {"type": "string", "enum": F.FACT_NAMES},
    "op": {"type": "string", "enum": [">=", ">", "<=", "<", "==", "!="]},
    "value": {"type": "number"}}, "required": ["fact", "op", "value"]}

TOOLS = [{"type": "function", "function": {
    "name": "propose_rule",
    "description": "Propose one improved detection rule.",
    "parameters": {"type": "object", "properties": {
        "change": {"type": "string", "maxLength": 240},
        "predicates": {"type": "array", "maxItems": MAX_PREDICATES, "items": {
            "type": "object", "properties": {
                "id": {"type": "string", "maxLength": 24},
                "clauses": {"type": "array", "maxItems": MAX_CLAUSES, "items": CLAUSE}},
            "required": ["id", "clauses"]}}},
        "required": ["change", "predicates"]}}}]

PROMPT = """You are tuning a bank's money-laundering detection rule.

The rule is evaluated in code against per-account-window facts. It fires if ANY predicate
matches; a predicate matches when ALL its clauses are true. There is no OR inside a
predicate -- express alternatives as separate predicates.

Base rate is about 1 window in 100. Precision is therefore hard and recall is cheap:
a rule that fires on everything gets perfect recall and is worthless. You are optimising F1.

FACT VOCABULARY -- these are the only facts that exist, with what they actually look like
in this population (p10 / median / p90 across all windows):
{vocab}

CURRENT RULE
{rule}

HOW IT SCORES on the development set
  precision {prec:.3f}   recall {rec:.3f}   F1 {f1:.3f}
  fires on {alerts} of {total} windows, {tp} of which are real ({planted} real in total)

PER-PREDICATE
{per_pred}

WHERE IT IS WRONG -- median fact values for the cases it MISSES versus the cases it
WRONGLY FLAGS. A fact where these two columns differ is a fact worth conditioning on.
{contrast}

{extra}
Propose ONE improved rule. You may add, remove, split, or retune predicates. Prefer a
small change you can justify from the numbers above over a rewrite. Keep at most {maxp}
predicates and {maxc} clauses each."""


def vocab_block(fx):
    out = []
    for name in F.FACT_NAMES:
        v = sorted(x[name] for x in fx.values())
        if not v:
            continue
        p = lambda q: v[min(len(v) - 1, int(q * len(v)))]
        out.append("  %-26s %10.4g %10.4g %10.4g" % (name, p(0.10), p(0.50), p(0.90)))
    return "\n".join(out)


def contrast_block(fx, fns, fps, top=12):
    """Facts ranked by how differently they behave on misses versus false alarms.
    This is the gradient information -- 20 raw rows would be the same signal, worse."""
    rows = []
    for name in F.FACT_NAMES:
        a = [fx[w][name] for w in fns]
        b = [fx[w][name] for w in fps]
        if not a or not b:
            continue
        sd = statistics.pstdev(a + b) or 1e-9
        rows.append((abs(statistics.median(a) - statistics.median(b)) / sd, name,
                     statistics.median(a), statistics.median(b)))
    rows.sort(reverse=True)
    out = ["  %-26s %12s %12s" % ("fact", "MISSED", "FALSE ALARM")]
    for _, name, ma, mb in rows[:top]:
        out.append("  %-26s %12.4g %12.4g" % (name, ma, mb))
    return "\n".join(out)


def per_predicate_block(rule, fx, wins, labels):
    out = []
    for p in rule["predicates"]:
        one = {"predicates": [p], "fire_if": "any"}
        hit = [w for w in wins if R.fires(one, fx[w])[0]]
        tp = len([w for w in hit if labels.get(w)])
        out.append("  %-24s fires %5d  real %3d  precision %.3f"
                   % (p["id"], len(hit), tp, tp / float(len(hit)) if hit else 0.0))
    return "\n".join(out) or "  (no predicates)"


def build_prompt(rule, fx, dev, labels, m, fns, fps, extra=""):
    return PROMPT.format(
        vocab=vocab_block(fx), rule=json.dumps(_to_clauses(rule), indent=1),
        prec=m["precision"], rec=m["recall"], f1=m["f1"], alerts=m["alerts"],
        total=len(dev), tp=m["tp"], planted=m["positives"],
        per_pred=per_predicate_block(rule, fx, dev, labels),
        contrast=contrast_block(fx, fns, fps), extra=extra,
        maxp=MAX_PREDICATES, maxc=MAX_CLAUSES)


def _to_clauses(rule):
    return {"predicates": [{"id": p["id"], "clauses": p["all"]} for p in rule["predicates"]]}


def validate(cand, fx_sample):
    """-> (rule, None) or (None, reason). Never raises, never coerces."""
    if not isinstance(cand, dict):
        return None, "not an object"
    preds = cand.get("predicates")
    if not isinstance(preds, list) or not preds:
        return None, "no predicates"
    if len(preds) > MAX_PREDICATES:
        return None, "%d predicates, max %d" % (len(preds), MAX_PREDICATES)
    out, seen = [], set()
    for i, p in enumerate(preds):
        cls = p.get("clauses") if isinstance(p, dict) else None
        if not isinstance(cls, list) or not cls:
            return None, "predicate %d has no clauses" % i
        if len(cls) > MAX_CLAUSES:
            return None, "predicate %d has %d clauses, max %d" % (i, len(cls), MAX_CLAUSES)
        norm = []
        for c in cls:
            if not isinstance(c, dict):
                return None, "clause is not an object"
            if c.get("fact") not in F.FACT_NAMES:
                return None, "unknown fact %r" % c.get("fact")
            if c.get("op") not in R.OPS:
                return None, "unknown op %r" % c.get("op")
            try:
                v = float(c["value"])
            except (TypeError, ValueError, KeyError):
                return None, "non-numeric value on %s" % c.get("fact")
            norm.append({"fact": c["fact"], "op": c["op"], "value": v})
        pid = str(p.get("id") or "p%d" % (i + 1))[:24]
        while pid in seen:
            pid += "'"
        seen.add(pid)
        out.append({"id": pid, "all": norm})
    rule = {"version": 0, "name": "candidate", "fire_if": "any", "predicates": out,
            "gloss": "", "change": str(cand.get("change", ""))[:240]}
    try:                                    # it must survive the evaluator, not just look valid
        for f in fx_sample:
            R.fires(rule, f)
    except Exception as e:
        return None, "evaluator raised: %s" % str(e)[:60]
    rule["gloss"] = R.gloss_from_json(rule)
    return rule, None


def propose(prompt, temperature=0.8, timeout=180):
    url, model = ENDPOINT
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "tools": TOOLS,
            "tool_choice": {"type": "function", "function": {"name": "propose_rule"}},
            "max_tokens": 800, "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    tc = d["choices"][0]["message"].get("tool_calls") or []
    if not tc:
        raise RuntimeError("no tool_call; finish=%s" % d["choices"][0]["finish_reason"])
    return json.loads(tc[0]["function"]["arguments"])
