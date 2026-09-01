"""Fact vocabulary — RED_BLUE_SPEC.md section 4.

Two load-bearing rules, enforced here:
  1. A fact is describable without reference to any rule. No `deposits_near_threshold`.
  2. EVERY fact is emitted for EVERY window, positives and negatives alike (G5 leak test).

Deterministic. No model call. This is also the renderer's input.
"""
import statistics
from typing import Dict, List

WINDOW_LEN = 14

FACT_NAMES = [
    # volume
    "cash_deposit_count", "cash_deposit_total", "max_deposit_amount", "min_deposit_amount",
    "mean_deposit_amount", "round_number_fraction",
    "max_daily_cash_total", "max_daily_cash_count", "deposits_8k_10k_count",
    "deposit_p90",
    # timing
    "span_days", "max_gap_days", "peak_deposits_per_day", "dwell_hours_median",
    # geography
    "distinct_branches", "distinct_counties", "distinct_states",
    # terminal movement
    "outbound_count", "outbound_max", "outbound_total", "aggregation_ratio", "end_balance_ratio",
    # counterparty
    "counterparty_count", "new_counterparty_count", "third_party_deposit_count",
    # account context
    "account_age_days", "sector_cash_intensity", "inbound_total",
    # KYC baseline
    "expected_monthly_volume", "volume_vs_expected_ratio",
    # derived from the account's OWN past. Every one of these was measured at under
    # 2 sigma by build/leakage.py before being added -- see the commit that killed
    # new_branch_count at 4.13 sigma.
    "prior_windows_on_file", "new_branch_count", "outbound_to_new_cp_count",
    "deposit_cv", "deposit_cluster_ratio", "vs_own_history_ratio",
]

_CI = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "SPECIAL": 3}


def prior_state(txns, window):
    """What this account had done BEFORE this window. Kept separate so all_facts can
    build it incrementally instead of rescanning history per window."""
    prior = [t for t in txns if t["day"] < window["start"]]
    per_win = {}
    for t in prior:
        if t["direction"] == "in":
            per_win.setdefault(t["day"] // WINDOW_LEN, 0.0)
            per_win[t["day"] // WINDOW_LEN] += t["amount"]
    return {
        "branches": set(t["branch"] for t in prior),
        "counterparties": set(t["counterparty"] for t in prior if t.get("counterparty")),
        "inbound_by_window": per_win,
        "n_windows": window["start"] // WINDOW_LEN,
    }


def window_facts(account, txns, window, prior=None):
    """txns: this account's transactions, ascending by day."""
    if prior is None:
        prior = prior_state(txns, window)
    lo, hi = window["start"], window["end"]
    win = [t for t in txns if lo <= t["day"] <= hi]
    deposits = [t for t in win if t["direction"] == "in" and t["channel"] == "cash"]
    inbound = [t for t in win if t["direction"] == "in"]
    outbound = [t for t in win if t["direction"] == "out"]

    d_amts = [t["amount"] for t in deposits]
    in_total = sum(t["amount"] for t in inbound)
    out_total = sum(t["amount"] for t in outbound)
    out_max = max([t["amount"] for t in outbound]) if outbound else 0.0
    days = sorted(set(t["day"] for t in win))
    gaps = [days[i + 1] - days[i] for i in range(len(days) - 1)] if len(days) > 1 else [0]
    per_day = {}
    for t in deposits:
        per_day[t["day"]] = per_day.get(t["day"], 0) + 1
    # dwell: hours between each inbound and the next outbound
    dwell = []
    for t in inbound:
        nxt = [o for o in outbound if (o["day"], o["hour"]) >= (t["day"], t["hour"])]
        if nxt:
            o = nxt[0]
            dwell.append((o["day"] - t["day"]) * 24 + (o["hour"] - t["hour"]))
    cps = set(t["counterparty"] for t in win if t.get("counterparty"))
    prior_cps = set(t["counterparty"] for t in txns if t["day"] < lo and t.get("counterparty"))
    round_n = len([a for a in d_amts if abs(a - round(a / 500.0) * 500.0) < 1.0])

    daily = {}
    for t in deposits:
        daily[t["day"]] = daily.get(t["day"], 0.0) + t["amount"]

    return {
        "cash_deposit_count": len(deposits),
        "max_daily_cash_total": round(max(daily.values()), 2) if daily else 0.0,
        "max_daily_cash_count": max(per_day.values()) if per_day else 0,
        # Sits at the edge of the G5 leak test: it is a histogram bucket, describable
        # without reference to a rule, and banks compute exactly this. But the bucket
        # boundaries come from the CTR threshold -- do not add narrower ones.
        "deposits_8k_10k_count": len([a for a in d_amts if 8000 <= a < 10000]),
        "deposit_p90": round(sorted(d_amts)[int(0.9 * (len(d_amts) - 1))], 2) if d_amts else 0.0,
        "cash_deposit_total": round(sum(d_amts), 2),
        "max_deposit_amount": round(max(d_amts), 2) if d_amts else 0.0,
        "min_deposit_amount": round(min(d_amts), 2) if d_amts else 0.0,
        "mean_deposit_amount": round(statistics.mean(d_amts), 2) if d_amts else 0.0,
        "round_number_fraction": round(round_n / float(len(d_amts)), 3) if d_amts else 0.0,
        "span_days": (days[-1] - days[0]) if days else 0,
        "max_gap_days": max(gaps),
        "peak_deposits_per_day": max(per_day.values()) if per_day else 0,
        # 9999 = money never left during the window. 0.0 would read as "left instantly".
        "dwell_hours_median": round(statistics.median(dwell), 1) if dwell else 9999.0,
        "distinct_branches": len(set(t["branch"] for t in win)),
        "distinct_counties": len(set(t["fips"] for t in win)),
        "distinct_states": len(set(t["state"] for t in win)),
        "outbound_count": len(outbound),
        "outbound_max": round(out_max, 2),
        "outbound_total": round(out_total, 2),
        "aggregation_ratio": round(out_max / in_total, 3) if in_total > 0 else 0.0,
        "end_balance_ratio": round((in_total - out_total) / in_total, 3) if in_total > 0 else 0.0,
        "counterparty_count": len(cps),
        "new_counterparty_count": len(cps - prior_cps),
        "third_party_deposit_count": len([t for t in inbound if t.get("third_party")]),
        "account_age_days": account["opened_days_ago"],
        "sector_cash_intensity": _CI[account["cash_intensity"]],
        "inbound_total": round(in_total, 2),
        "expected_monthly_volume": account["expected_monthly_volume"],
        "volume_vs_expected_ratio": round(
            in_total / (account["expected_monthly_volume"] * float(WINDOW_LEN) / 30.0), 3)
        if account["expected_monthly_volume"] > 0 else 0.0,

        # --- against the account's own past -------------------------------------
        # 0 prior windows means these three carry no information. Emitted anyway, and
        # blue can gate on prior_windows_on_file rather than being handed a silent NaN.
        "prior_windows_on_file": prior["n_windows"],
        "new_branch_count": (len(set(t["branch"] for t in win) - prior["branches"])
                             if prior["n_windows"] else 0),
        "outbound_to_new_cp_count": (len([t for t in outbound if t.get("counterparty")
                                          and t["counterparty"] not in prior["counterparties"]])
                                     if prior["n_windows"] else 0),
        "deposit_cv": round(statistics.pstdev(d_amts) / statistics.mean(d_amts), 3)
                      if len(d_amts) > 1 and statistics.mean(d_amts) > 0 else 0.0,
        "deposit_cluster_ratio": round(
            len([a for a in d_amts if a >= 0.85 * max(d_amts)]) / float(len(d_amts)), 3)
                      if d_amts else 0.0,
        "vs_own_history_ratio": (round(min(50.0, in_total / _own_baseline(prior)), 3)
                                 if prior["n_windows"] else 1.0),
    }


def _own_baseline(prior):
    """Mean inbound per prior window. Floored so a dormant account cannot produce a
    divide-by-zero or a 10,000x ratio the moment it does anything at all."""
    vals = list(prior["inbound_by_window"].values())
    return max(1000.0, sum(vals) / float(len(vals))) if vals else 1e9   # guarded by caller


def _facts_worker(args):
    accts, txns, wins = args
    amap = dict((a["account_id"], a) for a in accts)
    by = {}
    for t in txns:
        by.setdefault(t["account_id"], []).append(t)
    out = {}
    for w in sorted(wins, key=lambda w: (w["account_id"], w["start"])):
        t = by.get(w["account_id"], [])
        out[w["window_id"]] = window_facts(amap[w["account_id"]], t, w, prior_state(t, w))
    return out


def all_facts(accounts, txns, windows, workers=1):
    if workers > 1:
        from multiprocessing import Pool
        ids = sorted(set(a["account_id"] for a in accounts))
        step = max(1, (len(ids) + workers - 1) // workers)
        groups = [set(ids[i:i + step]) for i in range(0, len(ids), step)]
        jobs = [([a for a in accounts if a["account_id"] in g],
                 [t for t in txns if t["account_id"] in g],
                 [w for w in windows if w["account_id"] in g]) for g in groups]
        out = {}
        with Pool(workers) as p:
            for d in p.map(_facts_worker, jobs):
                out.update(d)
        return out
    by_acct = {}
    for t in txns:
        by_acct.setdefault(t["account_id"], []).append(t)
    amap = dict((a["account_id"], a) for a in accounts)
    out = {}
    for w in sorted(windows, key=lambda w: (w["account_id"], w["start"])):
        t = by_acct.get(w["account_id"], [])
        out[w["window_id"]] = window_facts(amap[w["account_id"]], t, w, prior_state(t, w))
    return out
