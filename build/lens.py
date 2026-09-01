"""The two panel lenses. Cascade: QWEN intent over every alert, NEMOTRON literal
over what Qwen flags plus the stratified audit sample.

Escalation is NOT a model decision -- it is qwen_verdict != nemo_verdict, in Python.
A model deciding "should I escalate" adds a third judgement with no ground truth.

Call config is the one BOX_STATE.md proved reliable: named tool choice + thinking
off + schema length caps. tool_choice:'required' lets the model ramble into prose
and return zero tool calls.
"""
import json, urllib.request
from typing import Dict, Optional

QWEN = ("http://localhost:8000/v1/chat/completions", "qwen")
NEMOTRON = ("http://localhost:8081/v1/chat/completions", "nemotron")

# The renderer deliberately never names the $10,000 line -- stating it there would
# hand over the comparison the rule already makes. It belongs here, as domain context.
CONTEXT = """You review alerts from a bank's anti-money-laundering transaction monitoring system.
Your job is to separate real laundering patterns from ordinary business that resembles them.
Both are common. Roughly one alert in thirty is real.

US banks must report any cash transaction over $10,000, so illicit cash is often kept below
that line, across days and branches.

PATTERNS THAT INDICATE LAUNDERING
- Structuring: many cash deposits clustered just under $10,000 -- not spread across a range,
  but bunched high and never crossing -- often at branches the account does not normally use.
- Funnel account: deposits at many branches, aggregated and sent out in one transfer.
- Layering: funds hopping between accounts with no economic purpose, in and straight out.
- Pass-through / mule: money in and out within days, balance returning to near zero, with
  counterparties that appear once and are never seen again.
- Cuckoo smurfing: third-party cash deposits into an account with no relationship to them.
- Velocity spike: sudden volume far above what the business has ever done, then back to normal.

ORDINARY BUSINESS THAT LOOKS LIKE THE ABOVE
- Cash-intensive trades -- restaurants, bars, salons, laundries -- bank large cash daily.
  About one US business in five. Their deposits spread ACROSS a range, tracking daily takings.
- Franchises deposit at several branches and sweep to one head-office account, every week,
  to the same counterparty.
- Payroll firms take many third-party deposits and pay them straight out.
- Escrow and title companies move large sums in and out within days by design.
- This bank spans Ohio, Kentucky and Indiana, which meet within twenty miles. Banking across
  two or three states is ordinary here and is not itself a signal.

The discriminator is rarely volume. It is whether the SHAPE of the activity is explained by
what the business is, AND whether it is a departure from what this account has always done.
A franchise sweeping to the same head-office account every week is not a funnel. A business
using branches it has never used before, or sending to a counterparty it has never dealt
with, is a different matter.
"""

INTENT = """{ctx}
Read the account-window below.

First describe the dominant pattern of money movement you actually see -- the shape, not a
summary. Then decide whether this business explains that shape.

Rule of thumb: heavy cash alone is not suspicious. Cash deposits BUNCHED just under $10,000,
or activity at branches the account is not registered to, or money that arrives and leaves
without doing anything, are.

{window}

Answer with the tool. Cite the specific transaction ids that drove your answer."""

LITERAL = """{ctx}
A detection rule flagged the account-window below.

THE RULE
{gloss}

IT FIRED ON: {predicate}

QUESTION: does the evidence actually support this rule firing, or is this a mechanical
match on an account where this behaviour is ordinary? You are checking the rule's work,
not re-deciding from scratch.

{window}

Answer with the tool. Cite the specific transaction ids that drove your answer."""

PROPS = {
    # generated BEFORE the verdict, so the model engages the rows first
    "pattern": {"type": "string", "maxLength": 220},
    "verdict": {"type": "string", "enum": ["suspicious", "benign"]},
    "typology": {"type": "string", "enum": ["structuring", "funnel", "layering",
                                            "pass_through", "cuckoo_smurfing",
                                            "velocity_spike", "none"]},
    "confidence": {"type": "number"},
    "rationale": {"type": "string", "maxLength": 400},
    "cited_txn_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
}
TOOLS = [{"type": "function", "function": {
    "name": "submit_verdict", "description": "Record the verdict for this account-window.",
    "parameters": {"type": "object", "properties": PROPS,
                   "required": ["pattern", "verdict", "typology", "confidence",
                                "rationale", "cited_txn_ids"]}}}]


def _call(endpoint, prompt, timeout=180):
    url, model = endpoint
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "tools": TOOLS,
            "tool_choice": {"type": "function", "function": {"name": "submit_verdict"}},
            "max_tokens": 400, "temperature": 0.1,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    tc = d["choices"][0]["message"].get("tool_calls") or []
    if not tc:
        raise RuntimeError("no tool_call; finish=%s" % d["choices"][0]["finish_reason"])
    return json.loads(tc[0]["function"]["arguments"])


def intent(window_text, timeout=180):
    return _call(QWEN, INTENT.format(ctx=CONTEXT, window=window_text), timeout=timeout)


def literal(window_text, gloss, predicate, timeout=180):
    return _call(NEMOTRON, LITERAL.format(ctx=CONTEXT, window=window_text,
                                          gloss=gloss, predicate=predicate),
                 timeout=timeout)
