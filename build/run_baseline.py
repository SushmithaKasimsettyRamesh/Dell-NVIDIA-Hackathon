"""Baseline run — does the corpus reproduce the 95% false-positive regime,
and is the resulting alert volume tractable for local models on the GB10?

    python3 build/run_baseline.py [--accounts N] [--base-rate R] [--seed S]
"""
import json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpus as C
import facts as F
import rules as R
import score as S

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# ---------------------------------------------------------------- MEASURED
# Replaced 22 Aug with the real numbers from BOX_STATE.md (80-call soak, concurrency 4,
# 0 failures). The earlier prefill/decode model was an estimate and was 6.2x optimistic.
# Re-measure with build/measure_throughput.py if anything about the serving stack changes.
CALLS_PER_S = 0.489            # MEASURED on promaxgb10-96ab, conc 4, p50 8.1s
MAX_CONCURRENCY = 4            # HARD CAP -- NVFP4 MoE MARLIN kernel deadlocks at 8.
                               # This is TOTAL in-flight across loop + panel + red,
                               # not 4 per stage. RED_BLUE_SPEC section 5.
PROMPT_TOKENS = 5000.0         # measured
OUTPUT_TOKENS = 234.0          # measured, with enable_thinking:false (1,262 if left on)
LENSES = 3
LOOP_BUDGET_S = 7200.0         # 2 hours, RACE_DAY.md


def calls_affordable(seconds=LOOP_BUDGET_S):
    return int(seconds * CALLS_PER_S)


def minutes(calls):
    return calls / CALLS_PER_S / 60.0


def arg(flag, default, cast=float):
    if flag in sys.argv:
        return cast(sys.argv[sys.argv.index(flag) + 1])
    return default


def main():
    n_acc = int(arg("--accounts", 400, float))
    base = arg("--base-rate", 0.01)
    seed = int(arg("--seed", 7, float))
    workers = int(arg("--workers", 1, float))

    t0 = time.time()
    c = C.Corpus(seed=seed, n_accounts=n_acc, base_rate=base, workers=workers).build()
    wins = c.windows()
    fx = F.all_facts(c.accounts, c.txns, wins, workers=workers)
    t_gen = time.time() - t0

    print("=" * 100)
    print("CORPUS  %d accounts | %d transactions | %d windows (%dd, %dd windows) | %.1fs"
          % (len(c.accounts), len(c.txns), len(wins), C.PERIOD_DAYS, C.WINDOW_DAYS, t_gen))
    pos = [w for w in wins if c.labels[w["window_id"]]]
    print("        %d planted typologies = %.2f%% base rate | %d facts/window, all windows"
          % (len(pos), 100.0 * len(pos) / len(wins), len(F.FACT_NAMES)))
    mix = {}
    for a in c.accounts:
        mix[a["cash_intensity"]] = mix.get(a["cash_intensity"], 0) + 1
    print("        cash intensity: " + " ".join(
        "%s %.1f%%" % (k, 100.0 * v / len(c.accounts)) for k, v in sorted(mix.items())))
    prof = {}
    for a in c.accounts:
        prof[a["profile"]] = prof.get(a["profile"], 0) + 1
    print("        profiles: " + " ".join("%s=%d" % (k, v) for k, v in sorted(prof.items())))
    print("=" * 100)

    results = {}
    for rule in (R.CTR_ONLY, R.BANK_SCENARIOS, R.SEED_RULE):
        pred = {}
        hits = {}
        t = time.time()
        for w in wins:
            ok, ids = R.fires(rule, fx[w["window_id"]])
            pred[w["window_id"]] = ok
            for i in ids:
                hits[i] = hits.get(i, 0) + 1
        el = time.time() - t
        m = S.metrics(pred, c.labels)
        m["elapsed_s"] = el
        m["predicate_hits"] = hits
        m["by_typology"] = S.recall_by_typology(pred, c.labels)
        results[rule["name"]] = m
        print(S.fmt(rule["name"], m))
        print("      recall by typology: " + "  ".join(
            "%s %d/%d" % (k, v[0], v[1]) for k, v in sorted(m["by_typology"].items())))
        if hits:
            print("      fired by predicate: " + ", ".join(
                "%s=%d" % (k, v) for k, v in sorted(hits.items())))
        print("      %d windows evaluated in %.3fs deterministically ($0, no model)"
              % (len(wins), el))
    print("=" * 100)

    bank = results[R.BANK_SCENARIOS["name"]]
    ctr = results[R.CTR_ONLY["name"]]

    print("\nWHAT THIS SHOWS")
    print("-" * 100)
    st = ctr["by_typology"].get("STRUCTURING", (0, 0))
    print("1. Literal $10k CTR rule:  overall recall %.3f, but STRUCTURING %d/%d."
          % (ctr["recall"], st[0], st[1]))
    print("   %s" % ("Confirms FALSE_POSITIVES.md's G9 warning: it catches only the typologies that\n"
                     "   do NOT structure. The few structuring hits come from the account's own benign\n"
                     "   cash deposits in the same window, not the planted sequence."
                     if st[0] <= 0.2 * max(1, st[1]) else
                     "STRUCTURING is leaking above $10k -- check the planter."))
    print("2. Bank scenario set:      FP rate %.1f%%  (industry cited: 95%%)  precision %.3f"
          % (100.0 * bank["fp_rate"], bank["precision"]))
    print("   specificity %.4f -> %s"
          % (bank["specificity"],
             "1 benign window in %d wrongly alerts" % int(round(1.0 / max(1e-9, 1 - bank["specificity"])))
             if bank["specificity"] < 1 else "no false alerts"))
    print("3. Headroom for the panel: %d alerts/period, %d of them true."
          % (bank["alerts"], bank["tp"]))

    print("\nWHAT THE DAY BUYS  (L0 rule gates, L1 panel judges -- RED_BLUE_SPEC section 2)")
    print("-" * 100)
    print("   MEASURED: %.3f calls/s at concurrency %d (BOX_STATE.md). %.0f prompt / %.0f completion tok."
          % (CALLS_PER_S, MAX_CONCURRENCY, PROMPT_TOKENS, OUTPUT_TOKENS))
    budget = calls_affordable()
    print("   2-hour loop budget = %d model calls TOTAL, across the loop AND red AND the panel.\n"
          % budget)
    print("   L0 scoring is DETERMINISTIC: %d windows in %.3fs, ZERO model calls, $0."
          % (len(wins), bank["elapsed_s"]))
    print("   => dev-set SIZE costs nothing. Scoring a rule is fires() over a fact dict.")
    print("      The loop's model calls are the REWRITES only, not the classifications.\n")

    ITERS, K = 100, 4
    rewrites, red = ITERS * K, ITERS * K
    sweep = bank["alerts"] * LENSES
    rows = [
        ("blue rewrites   %d iters x k=%d" % (ITERS, K), rewrites),
        ("red candidates  %d rounds x k=%d" % (ITERS, K), red),
        ("panel sweep     %d alerts x %d lenses" % (bank["alerts"], LENSES), sweep),
    ]
    print("   %-44s %8s %8s %8s" % ("component", "calls", "minutes", "of budget"))
    for name, calls in rows:
        print("   %-44s %8d %8.1f %7.0f%%" % (name, calls, minutes(calls), 100.0 * calls / budget))
    tot = rewrites + red + sweep
    print("   %-44s %8d %8.1f %7.0f%%  %s" % ("TOTAL", tot, minutes(tot), 100.0 * tot / budget,
          "FITS" if tot <= budget else "OVER BUDGET"))

    print("\n   The loop is cheap; the panel is the whole cost.")
    afford = int((budget - rewrites - red) / LENSES)
    print("   After %d rewrite + %d red calls, %d calls remain = %d alerts panelled ONCE."
          % (rewrites, red, budget - rewrites - red, afford))
    if bank["alerts"] > afford:
        tgt = int(n_acc * afford / float(bank["alerts"]))
        print("   You produce %d alerts. That is %.1fx too many." % (bank["alerts"], bank["alerts"] / float(afford)))
        print("   => run the corpus at ~%d accounts, or panel a stratified sample of %d alerts."
              % (tgt, afford))
    else:
        print("   You produce %d alerts -- fits with %d calls to spare."
              % (bank["alerts"], budget - tot))

    print("\n   Panel re-scored EVERY iteration (the version that would justify the box on")
    print("   volume alone): %d calls = %.0f hours. Not available at any corpus size."
          % (ITERS * sweep, minutes(ITERS * sweep) / 60.0))

    print("\n   🔴 CONCURRENCY IS THE BINDING CONSTRAINT, NOT TOKENS")
    print("   " + "-" * 78)
    print("   Total in-flight requests must stay <= %d. A 3-lens panel over k=4 candidates" % MAX_CONCURRENCY)
    print("   is 12 concurrent -> kernel deadlock -> container restart. Serialise the stages")
    print("   or share ONE ThreadPoolExecutor(max_workers=%d) across the whole engine." % MAX_CONCURRENCY)

    if bank["alerts"]:
        print("\n   Precision headroom: to reach 0.50 the panel must clear %.1f%% of the %d false"
              % (100.0 * (1.0 - bank["tp"] / float(max(1, bank["fp"]))), bank["fp"]))
        print("   alerts while keeping the %d true ones. L0 caps recall at %.3f."
              % (bank["tp"], bank["recall"]))

    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    with open(os.path.join(OUT, "accounts.json"), "w") as fh:
        json.dump(c.accounts, fh, indent=1)
    with open(os.path.join(OUT, "txns.jsonl"), "w") as fh:
        for t in c.txns:
            fh.write(json.dumps(t) + "\n")
    with open(os.path.join(OUT, "labels.jsonl"), "w") as fh:
        for w in wins:
            fh.write(json.dumps({"window_id": w["window_id"],
                                 "typology": c.labels[w["window_id"]]}) + "\n")
    with open(os.path.join(OUT, "facts.jsonl"), "w") as fh:
        for w in wins:
            fh.write(json.dumps({"window_id": w["window_id"], "facts": fx[w["window_id"]]}) + "\n")
    with open(os.path.join(OUT, "baseline_results.json"), "w") as fh:
        json.dump(dict((k, dict((kk, vv) for kk, vv in v.items())) for k, v in results.items()),
                  fh, indent=1)
    print("\nwrote build/out/{accounts.json,txns.jsonl,labels.jsonl,facts.jsonl,baseline_results.json}")


if __name__ == "__main__":
    main()
