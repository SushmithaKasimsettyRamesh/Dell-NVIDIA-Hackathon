"""Synthetic AML corpus generator — Census-anchored, stdlib only, Python 3.9.

Consumes build/data/business_mix.json (US Census CBP 2022):
  sector            <- national.sectors[].share          (8.3M establishments)
  home branch       <- bank_footprint.counties           (Cincinnati tri-state OH/KY/IN)
  cash behaviour    <- sectors[].cash_intensity          (~19% HIGH)

Emits accounts / transactions / 7-day windows / binary labels.
Ground truth is exact because the typologies are planted.
"""
import json, os, random
from typing import Dict, List, Optional

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "business_mix.json")

PERIOD_DAYS = 168          # 24 weeks
WINDOW_DAYS = 14           # a 7-day window cannot hold an "over 11 days" structuring case

# Benign profiles that superficially resemble a typology (PLAN.md 4b).
# weight is relative *within* a cash-intensity class.
BENIGN_PROFILES = {
    "HIGH":    [("HIGH_CASH", 6), ("CASH_HEAVY_BAR", 3), ("SEASONAL", 1)],
    "MEDIUM":  [("NORMAL", 5), ("FRANCHISE", 2), ("PAYROLL", 1), ("SEASONAL", 1)],
    "LOW":     [("NORMAL", 8), ("PAYROLL", 1)],
    "SPECIAL": [("REMITTANCE", 3), ("ESCROW", 2), ("NORMAL", 2)],
}

TYPOLOGIES = ["STRUCTURING", "FUNNEL", "PASS_THROUGH", "CUCKOO_SMURFING",
              "LAYERING", "VELOCITY_SPIKE"]


def _load_mix():
    with open(DATA) as fh:
        d = json.load(fh)
    return d["national"]["sectors"], d["bank_footprint"]["counties"]


def _weighted(rnd, items, key):
    total = sum(key(i) for i in items)
    x = rnd.random() * total
    acc = 0.0
    for i in items:
        acc += key(i)
        if x <= acc:
            return i
    return items[-1]


def _account_worker(args):
    """One process builds a slice of accounts. Independent by construction."""
    seed, lo, hi, n_total, base_rate = args
    c = Corpus(seed=seed, n_accounts=n_total, base_rate=base_rate)
    out = []
    for i in range(lo, hi):
        a = c._make_account(i, random.Random((seed << 20) ^ i))
        c.accounts = [a]
        c._amap = {a["account_id"]: a}
        c._tid = i * 100000
        c.txns = []
        c._benign(a, random.Random((seed << 21) ^ i))
        out.append((a, c.txns))
    return out


class Corpus(object):
    def __init__(self, seed=7, n_accounts=400, base_rate=0.01, workers=1):
        self.seed = seed
        self.workers = workers
        self.rnd = random.Random(seed)
        self.n_accounts = n_accounts
        self.base_rate = base_rate
        self.sectors, self.counties = _load_mix()
        self.branches = self._make_branches()
        self.accounts = []
        self.txns = []
        self.labels = {}          # window_id -> typology or None
        self._tid = 0
        self._dropped = set()

    # ---------- branch network ----------
    def _make_branches(self):
        out = []
        for c in self.counties:
            n = max(1, int(round(c["establishments"] / 4000.0)))
            for k in range(n):
                out.append({"branch": "B%02d" % (len(out) + 1),
                            "fips": c["fips"], "state": c["state"],
                            "county": c["name"], "weight": c["establishments"] / float(n)})
        return out

    # ---------- accounts ----------
    def _make_account(self, i, rnd=None):
        rnd = rnd or self.rnd
        sec = _weighted(rnd, self.sectors, lambda s: s["share"])
        ci = sec["cash_intensity"]
        opts = BENIGN_PROFILES[ci]
        profile = _weighted(rnd, opts, lambda o: o[1])[0]
        home = _weighted(rnd, self.branches, lambda b: b["weight"])
        # usual branches: home plus, for multi-site profiles, nearby ones (often cross-state here)
        k = {"FRANCHISE": 4, "PAYROLL": 3, "CASH_HEAVY_BAR": 2}.get(profile, 1)
        used = [home]
        while len(used) < k:
            b = _weighted(rnd, self.branches, lambda b: b["weight"])
            if b["branch"] not in [u["branch"] for u in used]:
                used.append(b)
        # Real businesses occasionally bank somewhere unfamiliar -- the owner travels,
        # a branch is closed, staff deposit near home. Without this, "used a branch it
        # never uses" separates planted from benign at 4 sigma BY CONSTRUCTION, and
        # blue would learn the generator instead of laundering.
        occasional = []
        while len(occasional) < 2:
            b = _weighted(rnd, self.branches, lambda b: b["weight"])
            if b["branch"] not in [u["branch"] for u in used]:
                occasional.append(b["branch"])
        relocate = rnd.randrange(20, PERIOD_DAYS - 20) if rnd.random() < 0.12 else None
        size = rnd.lognormvariate(0.0, 0.7)          # relative business size
        base_monthly = {"HIGH": 90000, "MEDIUM": 60000, "LOW": 45000, "SPECIAL": 120000}[ci]
        return {
            "account_id": "A%04d" % i,
            "sector": sec["sector"], "cash_intensity": ci, "profile": profile,
            "branches": [b["branch"] for b in used],
            "home_state": home["state"], "home_county": home["county"],
            "opened_days_ago": int(rnd.uniform(40, 3000)),
            "occasional_branches": occasional,
            "drift_p": rnd.uniform(0.03, 0.12),
            "relocate_day": relocate,          # ~12% open a location mid-period
            "expected_monthly_volume": round(base_monthly * size, 2),
            "counterparties": ["CP%05d" % rnd.randrange(1, 99999) for _ in range(rnd.randint(3, 12))],
            "primary_cp": "CP%05d" % rnd.randrange(1, 99999),
        }

    # ---------- transaction helpers ----------
    def _txn(self, acct, day, hour, amount, direction, channel, branch=None,
             counterparty=None, third_party=False):
        self._tid += 1
        b = branch or acct["branches"][0]
        meta = [x for x in self.branches if x["branch"] == b][0]
        self.txns.append({
            "txn_id": "T%06d" % self._tid, "account_id": acct["account_id"],
            "day": day, "hour": hour, "amount": round(amount, 2),
            "direction": direction, "channel": channel, "branch": b,
            "state": meta["state"], "fips": meta["fips"],
            "counterparty": counterparty, "third_party": third_party,
        })

    def _pick_branch(self, acct, day, rnd=None):
        rnd = rnd or self.rnd
        """Home branches, plus drift, plus a permanent new location for some accounts."""
        pool = list(acct["branches"])
        if acct["relocate_day"] is not None and day >= acct["relocate_day"]:
            pool = pool + [acct["occasional_branches"][0]]
        if rnd.random() < acct["drift_p"]:
            # anywhere in the network, not a fixed pair -- a fixed pair gets exhausted
            # after a few windows and the account never sees a novel branch again,
            # which is what left new_branch_count separating at 2.5 sigma.
            return _weighted(rnd, self.branches, lambda b: b["weight"])["branch"]
        return rnd.choice(pool)

    def _benign_cp(self, acct, rnd=None):
        """Mostly the standing relationship; sometimes a new vendor or one-off."""
        rnd = rnd or self.rnd
        if rnd.random() < 0.72:
            return acct["primary_cp"]
        return "CP%05d" % rnd.randrange(1, 99999)

    def _cash_deposit(self, acct, day, mu, sigma, branch=None, rnd=None):
        rnd = rnd or self.rnd
        amt = rnd.lognormvariate(mu, sigma)
        amt = max(120.0, min(amt, 48000.0))
        self._txn(acct, day, rnd.randint(9, 17), amt, "in", "cash",
                  branch or self._pick_branch(acct, day, rnd))

    # ---------- benign activity by profile ----------
    def _benign(self, acct, rnd=None):
        rnd = rnd or self.rnd
        p = acct["profile"]
        scale = acct["expected_monthly_volume"] / 60000.0
        for day in range(PERIOD_DAYS):
            dow = day % 7
            if p in ("HIGH_CASH", "CASH_HEAVY_BAR"):
                # daily-ish cash takings; bars run big enough to sit in the 8-10k band
                n = rnd.choice([1, 1, 1, 2]) if dow >= 4 else rnd.choice([0, 1, 1])
                mu = 8.45 if p == "CASH_HEAVY_BAR" else 7.55
                for _ in range(n):
                    self._cash_deposit(acct, day, mu + 0.35 * (scale - 1), 0.45, None, rnd)
            elif p == "PAYROLL":
                # client companies fund payroll: many third-party sub-10k deposits, then one sweep out
                if dow in (0, 3):
                    for _ in range(rnd.randint(2, 5)):
                        amt = min(rnd.lognormvariate(8.6, 0.5), 9900.0)
                        self._txn(acct, day, rnd.randint(9, 16), amt, "in", "check",
                                  self._pick_branch(acct, day, rnd),
                                  "CP%05d" % rnd.randrange(1, 99999), True)
                if dow == 4:
                    tot = sum(t["amount"] for t in self.txns
                              if t["account_id"] == acct["account_id"]
                              and t["direction"] == "in" and day - 6 <= t["day"] <= day)
                    if tot > 0:
                        self._txn(acct, day, 15, tot * rnd.uniform(0.88, 0.97), "out", "ach",
                                  acct["branches"][0], self._benign_cp(acct, rnd))
            elif p == "FRANCHISE":
                # deposits at several branches (often across state lines here), swept to HQ
                if dow < 6:
                    for b in acct["branches"]:
                        if rnd.random() < 0.55:
                            self._cash_deposit(acct, day, 7.9 + 0.3 * (scale - 1), 0.4, b, rnd)
                if dow == 6:
                    tot = sum(t["amount"] for t in self.txns
                              if t["account_id"] == acct["account_id"]
                              and t["direction"] == "in" and day - 6 <= t["day"] <= day)
                    if tot > 0:
                        self._txn(acct, day, 16, tot * rnd.uniform(0.85, 0.95), "out", "wire",
                                  acct["branches"][0], self._benign_cp(acct, rnd))
            elif p == "ESCROW":
                # large in, large out within 48h, balance returns to ~0
                if rnd.random() < 0.22:
                    amt = rnd.lognormvariate(11.2, 0.7)
                    cp = "CP%05d" % rnd.randrange(1, 99999)
                    self._txn(acct, day, 10, amt, "in", "wire", acct["branches"][0], cp, True)
                    self._txn(acct, min(PERIOD_DAYS - 1, day + rnd.randint(0, 2)), 14,
                              amt * rnd.uniform(0.97, 0.995), "out", "wire",
                              acct["branches"][0], "CP%05d" % rnd.randrange(1, 99999))
            elif p == "REMITTANCE":
                # MSB corridor: many small, many counterparties, steady outbound
                for _ in range(rnd.randint(0, 6)):
                    amt = rnd.lognormvariate(6.9, 0.6)
                    self._txn(acct, day, rnd.randint(9, 18), amt, "in", "cash",
                              self._pick_branch(acct, day, rnd),
                              "CP%05d" % rnd.randrange(1, 99999), True)
                if dow in (2, 5):
                    tot = sum(t["amount"] for t in self.txns
                              if t["account_id"] == acct["account_id"]
                              and t["direction"] == "in" and day - 3 <= t["day"] <= day)
                    if tot > 0:
                        self._txn(acct, day, 17, tot * rnd.uniform(0.8, 0.93), "out", "wire",
                                  acct["branches"][0], self._benign_cp(acct, rnd))
            elif p == "SEASONAL":
                peak = 1.0 + 2.4 * (1.0 if 55 <= day <= 75 else 0.0)
                if rnd.random() < 0.35 * peak:
                    self._cash_deposit(acct, day, 7.7 + 0.5 * (peak - 1), 0.5, None, rnd)
            else:  # NORMAL
                if rnd.random() < 0.30:
                    amt = rnd.lognormvariate(8.9, 0.8)
                    self._txn(acct, day, rnd.randint(9, 17), amt, "in",
                              rnd.choice(["ach", "check", "wire"]), acct["branches"][0],
                              rnd.choice(acct["counterparties"]))
                if rnd.random() < 0.25:
                    amt = rnd.lognormvariate(8.6, 0.8)
                    self._txn(acct, day, rnd.randint(9, 17), amt, "out",
                              rnd.choice(["ach", "check"]), acct["branches"][0],
                              rnd.choice(acct["counterparties"]))

    # ---------- planted typologies ----------
    def _launder_branches(self, acct, k):
        """Not every launderer is sophisticated. A third use their own branches because
        it is easier; a third mix; a third deliberately spread. Always-unfamiliar was a
        generator fingerprint, not a laundering signal."""
        rnd = self.rnd
        soph = rnd.random()
        own = list(acct["branches"])
        if soph < 0.35:
            return [rnd.choice(own) for _ in range(k)]
        far = self._branches_across_states(k)
        if soph < 0.70:
            mix = [rnd.choice(own)] + far[1:]
            rnd.shuffle(mix)
            return mix
        return far

    def _branches_across_states(self, k):
        """k branches spanning at least two states -- ordinary in this footprint."""
        rnd = self.rnd
        picks = [rnd.choice(self.branches)]
        while len(picks) < k:
            b = rnd.choice(self.branches)
            if b["branch"] in [p["branch"] for p in picks]:
                continue
            picks.append(b)
        if len(set(p["state"] for p in picks)) < 2:
            other = [b for b in self.branches if b["state"] != picks[0]["state"]]
            picks[-1] = rnd.choice(other)
        return [p["branch"] for p in picks]

    def _plant(self, acct, start, typ, end=None):
        """All planted transactions stay inside [start, end] -- otherwise the label
        points at a window that does not contain the behaviour."""
        rnd = self.rnd
        brs = self.branches
        end = end if end is not None else start + WINDOW_DAYS - 1
        clamp = lambda d: max(start, min(int(d), end))
        before = self._tid
        if typ == "STRUCTURING":
            picks = self._launder_branches(acct, 3)
            n = rnd.randint(6, 9)
            # distinct days: a same-day cash aggregate over $10k files a CTR, which is
            # exactly what the launderer is avoiding.
            days = sorted(rnd.sample(range(start, min(end, start + 13) + 1), min(n, end - start + 1)))
            tot = 0.0
            for i, d in enumerate(days):
                # soft edges, and 15% of the time the launderer is careless
                if rnd.random() < 0.15:
                    amt = rnd.uniform(4800, 9880)
                else:
                    amt = min(9880.0, max(7600.0, rnd.gauss(9050, 780)))
                tot += amt
                self._txn(acct, d, rnd.randint(9, 17), amt, "in", "cash", picks[i % 3])
            self._txn(acct, end - rnd.randint(0, 1), 16, tot * rnd.uniform(0.86, 0.95), "out", "wire",
                      picks[0], "CP%05d" % rnd.randrange(1, 99999))
        elif typ == "FUNNEL":
            picks = self._launder_branches(acct, 5)
            tot = 0.0
            for i in range(rnd.randint(8, 14)):
                amt = rnd.uniform(2500, 9400)
                tot += amt
                self._txn(acct, clamp(start + rnd.randint(0, WINDOW_DAYS - 3)), rnd.randint(9, 18), amt,
                          "in", "cash", picks[i % 5], None, True)
            self._txn(acct, end, 17, tot * rnd.uniform(0.9, 0.97), "out", "wire",
                      picks[0], "CP%05d" % rnd.randrange(1, 99999))
        elif typ == "PASS_THROUGH":
            for i in range(rnd.randint(2, 4)):
                amt = rnd.uniform(40000, 120000)
                cp = "CP%05d" % rnd.randrange(1, 99999)
                d = clamp(start + i * 2)
                self._txn(acct, d, 10, amt, "in", "wire", acct["branches"][0], cp, True)
                self._txn(acct, clamp(d + 1), 15, amt * rnd.uniform(0.985, 0.999),
                          "out", "wire", acct["branches"][0], "CP%05d" % rnd.randrange(1, 99999))
        elif typ == "CUCKOO_SMURFING":
            for i in range(rnd.randint(5, 9)):
                self._txn(acct, clamp(start + rnd.randint(0, WINDOW_DAYS - 1)), rnd.randint(9, 18),
                          rnd.uniform(1800, 8800), "in", "cash",
                          self._launder_branches(acct, 3)[i % 3],
                          "CP%05d" % rnd.randrange(1, 99999), True)
        elif typ == "LAYERING":
            for i in range(rnd.randint(6, 10)):
                amt = rnd.uniform(15000, 45000)
                d = clamp(start + rnd.randint(0, WINDOW_DAYS - 1))
                self._txn(acct, d, 11, amt, "in", "ach", acct["branches"][0],
                          "CP%05d" % rnd.randrange(1, 400), False)
                self._txn(acct, d, 13, amt * rnd.uniform(0.97, 0.999), "out", "ach",
                          acct["branches"][0], "CP%05d" % rnd.randrange(1, 400))
        elif typ == "VELOCITY_SPIKE":
            for i in range(rnd.randint(10, 18)):
                self._cash_deposit(acct, clamp(start + rnd.randint(0, WINDOW_DAYS - 1)), 9.05, 0.3)
        return set("T%06d" % i for i in range(before + 1, self._tid + 1))

    # ---------- windows ----------
    def windows(self):
        out = []
        n_win = PERIOD_DAYS // WINDOW_DAYS
        for a in self.accounts:
            for w in range(n_win):
                out.append({"window_id": "%s-W%02d" % (a["account_id"], w),
                            "account_id": a["account_id"],
                            "start": w * WINDOW_DAYS, "end": (w + 1) * WINDOW_DAYS - 1})
        return out

    def build(self):
        if self.workers > 1:
            return self._build_parallel()
        self.accounts = [self._make_account(i, random.Random((self.seed << 20) ^ i))
                         for i in range(1, self.n_accounts + 1)]
        for i, a in enumerate(self.accounts, 1):
            self._benign(a, random.Random((self.seed << 21) ^ i))
        return self._plant_all()

    def _plant_all(self):
        wins = self.windows()
        for w in wins:
            self.labels[w["window_id"]] = None
        n_pos = max(1, int(round(len(wins) * self.base_rate)))
        # plant inside the population that legitimately looks like this (HIGH-cash + SPECIAL)
        CASH_TYPOLOGIES = ("STRUCTURING", "FUNNEL", "CUCKOO_SMURFING", "VELOCITY_SPIKE")

        def eligible(a, typ):
            # RED_BLUE_SPEC section 5 constraint 2: the sequence must be plausible for the
            # sector. A dental practice does not take nine cash deposits a week.
            if typ in CASH_TYPOLOGIES:
                return a["cash_intensity"] in ("HIGH", "SPECIAL")
            return True

        pool = [w for w in wins if w["start"] > 0]
        self.rnd.shuffle(pool)
        used_accounts = set()
        planted = 0
        # typology chosen first, then an eligible account found for it -- otherwise the
        # eligibility filter silently skews the mix toward the non-cash typologies.
        while planted < n_pos:
            typ = TYPOLOGIES[planted % len(TYPOLOGIES)]
            w = None
            for cand in pool:
                if cand["account_id"] in used_accounts:
                    continue
                if eligible(self._acct(cand["account_id"]), typ):
                    w = cand
                    break
            if w is None:
                break
            planted_ids = self._plant(self._acct(w["account_id"]), w["start"], typ, w["end"])
            # A front business COMMINGLES: the laundered cash is banked AS the takings,
            # displacing genuine ones, so volume barely moves. Planting that always adds
            # on top left 90% of planted windows above 1.5x their own history against 15%
            # of benign -- laundering the generator could only ever do one way.
            if typ in CASH_TYPOLOGIES and self.rnd.random() < 0.45:
                self._commingle(w, planted_ids)
            self.labels[w["window_id"]] = typ
            used_accounts.add(w["account_id"])
            pool.remove(w)
            planted += 1
        if self._dropped:
            self.txns = [t for t in self.txns if t["txn_id"] not in self._dropped]
        self.txns.sort(key=lambda t: (t["account_id"], t["day"], t["hour"]))
        return self

    def _commingle(self, w, planted_ids):
        """Drop most of the window's genuine cash takings -- the laundered cash stands
        in for them. Leaves the pattern intact and the volume signature flat."""
        rnd = self.rnd
        genuine = [t for t in self.txns
                   if t["account_id"] == w["account_id"]
                   and w["start"] <= t["day"] <= w["end"]
                   and t["direction"] == "in" and t["channel"] == "cash"
                   and t["txn_id"] not in planted_ids]
        rnd.shuffle(genuine)
        self._dropped.update(t["txn_id"]
                             for t in genuine[:int(len(genuine) * rnd.uniform(0.55, 0.9))])

    def _build_parallel(self):
        """Same output as workers=1 -- each account has its own deterministic stream."""
        from multiprocessing import Pool
        n, w = self.n_accounts, self.workers
        step = max(1, (n + w - 1) // w)
        jobs = [(self.seed, lo + 1, min(n, lo + step) + 1, n, self.base_rate)
                for lo in range(0, n, step)]
        with Pool(w) as p:
            for chunk in p.map(_account_worker, jobs):
                for a, txns in chunk:
                    self.accounts.append(a)
                    self.txns.extend(txns)
        self._plant_all()
        return self

    def branch_map(self):
        """branch_id -> county/state, for the renderer."""
        return dict((b["branch"], {"county": b["county"], "state": b["state"], "fips": b["fips"]})
                    for b in self.branches)

    def _acct(self, aid):
        if not hasattr(self, "_amap"):
            self._amap = dict((a["account_id"], a) for a in self.accounts)
        return self._amap[aid]
