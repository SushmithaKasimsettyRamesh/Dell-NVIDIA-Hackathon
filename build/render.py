"""Account-window -> prose + citable rows. DETERMINISTIC TEMPLATING, never an LLM call.

Two rules, both load-bearing (RED_BLUE_SPEC section 4, OPEN_QUESTIONS G5):

  1. EVERY slot is emitted for EVERY window. A sentence that appears only when it is
     incriminating IS the label. "No cash deposits in this window" is a slot, not an omission.

  2. Expose what the PREDICATE VOCABULARY CANNOT. The rule already keys on
     deposits_8k_10k_count -- so if the prose says "seven deposits between $8,000 and
     $10,000" the panel is re-deriving the rule in English and cannot add anything.
     State the amounts; let the model notice the clustering. Never name the $10,000
     threshold here: the lens system prompt carries it as domain context.

The renderer does ARITHMETIC. The model does INFERENCE.
"""
from typing import Dict, List, Optional

DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MAX_ROWS = 35
WINDOW_LEN = 14
CHANNEL = {"cash": "cash in", "check": "check in", "ach": "ACH in", "wire": "wire in"}
CHANNEL_OUT = {"cash": "cash out", "check": "check out", "ach": "ACH out", "wire": "wire out"}


def _money(x):
    return "$%s" % format(int(round(x)), ",d")


def _n(x):
    return "no" if x == 0 else str(x)


def account_block(a):
    return ("Account %s. %s business, cash-intensity %s. Opened %d days ago. "
            "Expected inbound volume %s per month. Registered to %d branch%s: %s."
            % (a["account_id"], a["sector"], a["cash_intensity"], a["opened_days_ago"],
               _money(a["expected_monthly_volume"]), len(a["branches"]),
               "" if len(a["branches"]) == 1 else "es", ", ".join(a["branches"])))


def narrative(a, f, win, branch_map, txns_in_window):
    """Seven slots, fixed order, always all seven."""
    s = []
    lo, hi = win["start"], win["end"]
    span = hi - lo + 1

    # 1. cash deposits
    if f["cash_deposit_count"] == 0:
        s.append("No cash deposits in this %d-day window." % span)
    else:
        s.append("%s cash deposit%s totalling %s, ranging %s to %s."
                 % (_n(f["cash_deposit_count"]),
                    "" if f["cash_deposit_count"] == 1 else "s",
                    _money(f["cash_deposit_total"]),
                    _money(f["min_deposit_amount"]), _money(f["max_deposit_amount"])))

    # 2. rhythm
    if f["cash_deposit_count"] == 0:
        s.append("No deposit activity to time.")
    else:
        days = sorted(set(t["day"] for t in txns_in_window
                          if t["direction"] == "in" and t["channel"] == "cash"))
        s.append("They fell on %s separate day%s spanning %d days, at most %s on any one day, "
                 "with the longest gap %d day%s."
                 % (_n(len(days)), "" if len(days) == 1 else "s", f["span_days"],
                    _n(f["max_daily_cash_count"]), f["max_gap_days"],
                    "" if f["max_gap_days"] == 1 else "s"))

    # 3. geography
    states = sorted(set(t["state"] for t in txns_in_window))
    counties = sorted(set(branch_map[t["branch"]]["county"] for t in txns_in_window))
    if not txns_in_window:
        s.append("No branch activity.")
    else:
        s.append("Activity ran through %s branch%s in %s (%s), across %s count%s."
                 % (_n(f["distinct_branches"]), "" if f["distinct_branches"] == 1 else "es",
                    "%s state%s" % (_n(f["distinct_states"]),
                                    "" if f["distinct_states"] == 1 else "s"),
                    ", ".join(states), _n(f["distinct_counties"]),
                    "y" if f["distinct_counties"] == 1 else "ies"))

    # 4. other inbound
    other = [t for t in txns_in_window if t["direction"] == "in" and t["channel"] != "cash"]
    if not other:
        s.append("No non-cash deposits.")
    else:
        s.append("Also %s non-cash deposit%s totalling %s, of which %s came from third parties."
                 % (_n(len(other)), "" if len(other) == 1 else "s",
                    _money(sum(t["amount"] for t in other)),
                    _n(f["third_party_deposit_count"])))

    # 5. outbound
    if f["outbound_count"] == 0:
        s.append("Nothing left the account during this window.")
    else:
        outs = [t for t in txns_in_window if t["direction"] == "out"]
        big = max(outs, key=lambda t: t["amount"])
        if f["outbound_count"] == 1:
            s.append("One outbound transfer of %s by %s on day %d of the window, to %s."
                     % (_money(big["amount"]), big["channel"], big["day"] - lo + 1,
                        big.get("counterparty") or "an unnamed party"))
        else:
            s.append("%s outbound transfers totalling %s; the largest was %s by %s on day %d "
                     "of the window." % (_n(f["outbound_count"]), _money(f["outbound_total"]),
                                         _money(big["amount"]), big["channel"],
                                         big["day"] - lo + 1))

    # 6. balance behaviour
    if f["inbound_total"] <= 0:
        s.append("No inbound funds to track.")
    else:
        s.append("%d%% of the money that came in was still in the account at the end of the window."
                 % int(round(100 * f["end_balance_ratio"])))

    # 7. against the account's own baseline
    s.append("Counterparties seen: %s, of which %s appear%s here for the first time. "
             "Inbound was %.1f times this account's expected volume for a period of this length."
             % (_n(f["counterparty_count"]), _n(f["new_counterparty_count"]),
                "s" if f["new_counterparty_count"] == 1 else "",
                f["volume_vs_expected_ratio"]))
    return " ".join(s)


def records(txns_in_window, win, branch_map):
    """Grouped by day, weekday named, ids for citation. The PATTERN lives here --
    aggregates dilute it. Deterministic cap, oldest dropped, and it says so."""
    rows = sorted(txns_in_window, key=lambda t: (t["day"], t["hour"]))
    dropped = 0
    if len(rows) > MAX_ROWS:
        dropped = len(rows) - MAX_ROWS
        rows = rows[dropped:]
    out, shown, last = [], [], None
    for t in rows:
        d = t["day"]
        head = ""
        if d != last:
            head = "Day %-3d %s" % (d - win["start"] + 1, DOW[d % 7])
            last = d
        b = branch_map[t["branch"]]
        label = (CHANNEL if t["direction"] == "in" else CHANNEL_OUT)[t["channel"]]
        cp = ""
        if t.get("counterparty"):
            cp = "  -> %s" % t["counterparty"]
        out.append("  %-12s %s  %-9s %12s   %s (%s, %s)%s"
                   % (head, t["txn_id"], label, _money(t["amount"]),
                      t["branch"], b["county"], b["state"], cp))
        shown.append(t["txn_id"])
    note = ""
    if dropped:
        note = "  [%d earlier transactions in this window not shown]\n" % dropped
    return note + "\n".join(out), shown


def history(a, txns, win, branch_map):
    """The account's own past, which is what actually separates a franchise from a
    funnel: the franchise sweeps to the SAME counterparty every week. Without this
    the first window of an account reports "1 counterparty, 1 new" as an artifact of
    having no history, and the model reads that as a fresh mule account."""
    prior = [t for t in txns if t["day"] < win["start"]]
    if not prior:
        return ("No prior history on file for this account -- this is the earliest window "
                "on record, so nothing here can be compared against its own past.")
    n_win = max(1, (win["start"] + WINDOW_LEN - 1) // WINDOW_LEN)
    dep = [t for t in prior if t["direction"] == "in" and t["channel"] == "cash"]
    outs = [t for t in prior if t["direction"] == "out"]
    inw = [t for t in txns if win["start"] <= t["day"] <= win["end"]]
    now_cps = set(t["counterparty"] for t in inw if t.get("counterparty"))
    past_cps = set(t["counterparty"] for t in prior if t.get("counterparty"))
    repeat = now_cps & past_cps
    past_branches = sorted(set(t["branch"] for t in prior))
    new_branches = sorted(set(t["branch"] for t in inw) - set(past_branches))

    s = ["Over the preceding %d comparable periods this account averaged %d cash deposits "
         "totalling %s per period, and %d outbound transfers."
         % (n_win, len(dep) / n_win, _money(sum(t["amount"] for t in dep) / n_win),
            len(outs) / n_win)]
    s.append("It normally uses %d branch(es): %s."
             % (len(past_branches), ", ".join(past_branches)))
    if new_branches:
        s.append("Branches used in this window that it has NEVER used before: %s."
                 % ", ".join(new_branches))
    else:
        s.append("Every branch used in this window is one it has used before.")
    if repeat:
        s.append("Counterparties in this window it has dealt with before: %s."
                 % ", ".join(sorted(repeat)))
    else:
        s.append("It has dealt with none of this window's counterparties before.")
    return " ".join(s)


def render_window(a, txns, win, f, branch_map):
    inw = [t for t in txns if win["start"] <= t["day"] <= win["end"]]
    rec, shown = records(inw, win, branch_map)
    return {
        "window_id": win["window_id"],
        "account": account_block(a),
        "history": history(a, txns, win, branch_map),
        "narrative": narrative(a, f, win, branch_map, inw),
        "records": rec,
        "citable_ids": shown,
        "n_txns": len(inw),
    }


def as_prompt(r):
    return ("ACCOUNT\n%s\n\nTHIS ACCOUNT'S OWN HISTORY\n%s\n\nWHAT HAPPENED\n%s\n\n"
            "TRANSACTIONS (cite these ids)\n%s\n"
            % (r["account"], r["history"], r["narrative"], r["records"]))
