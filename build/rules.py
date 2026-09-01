"""Rule schema + evaluator — RED_BLUE_SPEC.md section 3.

A rule is JSON predicates over named facts, plus a prose gloss the panel reads.
Deterministic, ~0 ms, free. This is L0: what RED attacks and BLUE rewrites.
"""
OPS = {">=": lambda a, b: a >= b, ">": lambda a, b: a > b,
       "<=": lambda a, b: a <= b, "<": lambda a, b: a < b,
       "==": lambda a, b: a == b, "!=": lambda a, b: a != b}


def fires(rule, facts):
    """-> (bool, [predicate ids that fired])  — which one fired IS the audit trail."""
    hits = []
    for p in rule["predicates"]:
        ok = True
        for c in p["all"]:
            v = facts.get(c["fact"])
            if v is None or not OPS[c["op"]](v, c["value"]):
                ok = False
                break
        if ok:
            hits.append(p["id"])
    return (len(hits) > 0, hits)


# ---------------------------------------------------------------- baselines

# (1) The literal CTR threshold. FALSE_POSITIVES.md predicts ~0 recall: every planted
#     structuring case is under $10k by construction. Included to SHOW that, not to use it.
CTR_ONLY = {
    "version": 0, "name": "CTR $10,000 threshold (literal)",
    "gloss": "File a report when a single cash deposit meets or exceeds $10,000.",
    "fire_if": "any",
    "predicates": [
        {"id": "ctr", "all": [{"fact": "max_deposit_amount", "op": ">=", "value": 10000}]},
    ],
}

# (2) What a bank actually runs: an accumulated scenario set. This is the honest G9
#     comparator — FALSE_POSITIVES.md Reason 8, "nothing is ever retired".
BANK_SCENARIOS = {
    "version": 1, "name": "Bank scenario set (4 accumulated scenarios)",
    "gloss": ("Fires when any of: cash deposits aggregate over $10,000 in one business day "
              "above the account's expected profile; "
              "three or more deposits between $8,000 and $10,000; funds in and out within "
              "72 hours leaving a near-zero balance; or deposits at two or more branches "
              "within the window followed by an outbound transfer."),
    "fire_if": "any",
    "predicates": [
        {"id": "s1_daily_aggregate_cash", "all": [
            {"fact": "max_daily_cash_total", "op": ">", "value": 10000},
            {"fact": "volume_vs_expected_ratio", "op": ">=", "value": 1.0}]},
        {"id": "s2_structuring_band", "all": [
            {"fact": "deposits_8k_10k_count", "op": ">=", "value": 3},
            {"fact": "volume_vs_expected_ratio", "op": ">=", "value": 1.0}]},
        {"id": "s3_rapid_movement", "all": [
            {"fact": "dwell_hours_median", "op": "<=", "value": 72},
            {"fact": "outbound_count", "op": ">=", "value": 1},
            {"fact": "end_balance_ratio", "op": "<=", "value": 0.1},
            {"fact": "inbound_total", "op": ">=", "value": 25000},
            {"fact": "volume_vs_expected_ratio", "op": ">=", "value": 1.0}]},
        {"id": "s4_multi_branch", "all": [
            {"fact": "distinct_branches", "op": ">=", "value": 3},
            {"fact": "distinct_states", "op": ">=", "value": 2},
            {"fact": "cash_deposit_count", "op": ">=", "value": 3},
            {"fact": "outbound_count", "op": ">=", "value": 1},
            {"fact": "volume_vs_expected_ratio", "op": ">=", "value": 1.0}]},
    ],
}

# (3) A seed for BLUE to start from Saturday — deliberately mediocre, one predicate.
SEED_RULE = {
    "version": 1, "name": "seed",
    "gloss": ("Three or more cash deposits under $10,000 within the window at two or more "
              "branches, followed by an outbound transfer of 80% or more of the total."),
    "fire_if": "any",
    "predicates": [
        {"id": "p1", "all": [
            {"fact": "cash_deposit_count", "op": ">=", "value": 3},
            {"fact": "max_deposit_amount", "op": "<", "value": 10000},
            {"fact": "distinct_branches", "op": ">=", "value": 2},
            {"fact": "aggregation_ratio", "op": ">=", "value": 0.4}]},
    ],
}


# ---------------------------------------------------------------- gloss
# BLUE rewrites the JSON; the gloss is REGENERATED from it and handed to the literal
# lens. A hand-written gloss would drift out of sync with the predicates the moment
# blue edits them, and the literal lens would be auditing a rule that no longer exists.

PHRASE = {
    "cash_deposit_count":       ("at least %s cash deposits", "fewer than %s cash deposits"),
    "cash_deposit_total":       ("cash deposits totalling over %s", "cash deposits totalling under %s"),
    "max_daily_cash_total":     ("more than %s in cash on a single day", "under %s in cash on any day"),
    "max_deposit_amount":       ("a single cash deposit of %s or more", "no single cash deposit reaching %s"),
    "min_deposit_amount":       ("every cash deposit at least %s", "some cash deposit below %s"),
    "deposits_8k_10k_count":    ("at least %s deposits in the $8,000-$10,000 band", "fewer than %s such deposits"),
    "distinct_branches":        ("deposits at %s or more branches", "deposits at fewer than %s branches"),
    "distinct_states":          ("activity in %s or more states", "activity in fewer than %s states"),
    "span_days":                ("spanning at least %s days", "spanning under %s days"),
    "outbound_count":           ("at least %s outbound transfers", "fewer than %s outbound transfers"),
    "aggregation_ratio":        ("a single outbound transfer of at least %s of money in",
                                 "no outbound transfer reaching %s of money in"),
    "end_balance_ratio":        ("more than %s of inbound funds remaining",
                                 "at most %s of inbound funds remaining"),
    "dwell_hours_median":       ("funds held longer than %s hours", "funds leaving within %s hours"),
    "volume_vs_expected_ratio": ("volume at least %sx the account's expected profile",
                                 "volume under %sx the expected profile"),
    "vs_own_history_ratio":     ("volume at least %sx what this account normally does",
                                 "volume under %sx its own normal"),
    "new_branch_count":         ("at least %s branches this account has never used before",
                                 "fewer than %s previously-unused branches"),
    "outbound_to_new_cp_count": ("at least %s transfers to counterparties never dealt with before",
                                 "fewer than %s transfers to new counterparties"),
    "deposit_cv":               ("cash deposit amounts varying by at least %s of the mean",
                                 "cash deposits bunched within %s of the mean"),
    "deposit_cluster_ratio":    ("at least %s of deposits sitting near the largest one",
                                 "under %s of deposits near the largest"),
    "prior_windows_on_file":    ("at least %s earlier periods of history on file",
                                 "fewer than %s earlier periods on file"),
    "inbound_total":            ("total inbound of %s or more", "total inbound under %s"),
    "third_party_deposit_count": ("at least %s third-party deposits", "fewer than %s third-party deposits"),
    "new_counterparty_count":   ("at least %s counterparties seen for the first time",
                                 "fewer than %s new counterparties"),
}
MONEY_FACTS = ("cash_deposit_total", "max_daily_cash_total", "max_deposit_amount",
               "min_deposit_amount", "inbound_total")
RATIO_FACTS = ("aggregation_ratio", "end_balance_ratio", "deposit_cv",
               "deposit_cluster_ratio")


def _val(fact, v):
    if fact == "volume_vs_expected_ratio" and abs(v - 1.0) < 1e-9:
        return "1"
    if fact in MONEY_FACTS:
        return "$%s" % format(int(v), ",d")
    if fact in RATIO_FACTS:
        return "%d%%" % int(round(100 * v))
    return str(v)


import re

_SING = [("counterparties", "counterparty"), ("branches", "branch"), ("counties", "county")]


def _singularise(out):
    """This text goes straight into a lens prompt -- "at least 1 outbound transfers"
    is noise the model has to parse around."""
    for a, b in _SING:
        out = re.sub(r"at least 1 (.*?)" + a + r"\b", r"at least 1 \1" + b, out)
    return re.sub(r"at least 1 (.+?)s\b", r"at least 1 \1", out)


def _clause(c):
    up = c["op"] in (">=", ">")
    pair = PHRASE.get(c["fact"])
    if not pair:
        return "%s %s %s" % (c["fact"], c["op"], c["value"])
    return _singularise(pair[0 if up else 1] % _val(c["fact"], c["value"]))


def gloss_from_json(rule):
    """Prose the literal lens can actually audit against."""
    parts = []
    for p in rule["predicates"]:
        parts.append("(%s) %s" % (p["id"], ", and ".join(_clause(c) for c in p["all"])))
    if len(parts) == 1:
        return "The rule fires when there is " + parts[0].split(") ", 1)[1] + "."
    return ("The rule fires when ANY of the following holds:\n  - "
            + "\n  - ".join(parts))


def predicate_gloss(rule, pid):
    for p in rule["predicates"]:
        if p["id"] == pid:
            return ", and ".join(_clause(c) for c in p["all"])
    return pid
