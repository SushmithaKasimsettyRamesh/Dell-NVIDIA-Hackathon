#!/usr/bin/env python3
"""
Throughput calibration for the vLLM / llama.cpp endpoint.

Run this the moment the box is serving. It answers the only question that matters
at 11:00: HOW MANY CALLS CAN THE LOOP AFFORD?

Measures calls/sec at several concurrency levels, finds where scaling flattens,
and prints the loop-sizing arithmetic backwards from your time budget.

Python 3.9 · stdlib only · no pip, no network beyond the local endpoint.

    python3 measure_throughput.py --model nvidia/Qwen3.6-35B-A3B-NVFP4
    python3 measure_throughput.py --url http://localhost:8081/v1/chat/completions \
        --model nemotron --levels 1,2,4 --budget 7200

Why it varies the prompt on every call: vLLM does automatic prefix caching.
Send the same prompt 32 times and you measure the cache, not the model, and the
number will be gloriously wrong in the direction you want to believe.
"""

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

# Roughly the shape of a real classify call: taxonomy + rule + narrative + rows.
# Padding is varied per call so no two requests share a prefix.
PREAMBLE = (
    "You are a transaction monitoring analyst. Apply the rule below exactly as written "
    "and return a verdict of SUSPICIOUS or CLEAR with the transaction ids you relied on.\n"
    "RULE: three or more cash deposits under $10,000 within seven days at two or more "
    "branches, followed by an outbound transfer of 80% or more of the deposited total.\n"
)

FILLER = (
    "Account {i} is a {sector} business opened {age} days ago in {county} County. "
    "Over the window it received {n} cash deposits totalling ${total:,}, the largest "
    "${mx:,}, across {br} branches in {st} states, followed by {ob} outbound transfer(s). "
    "No single deposit met the $10,000 CTR threshold. Transaction ids: {ids}. "
)

SECTORS = ["restaurant", "salon", "laundromat", "consulting", "auto repair",
           "convenience store", "landscaping", "dental practice"]
COUNTIES = ["Hamilton", "Butler", "Warren", "Clermont", "Kenton", "Boone",
            "Campbell", "Dearborn"]


# Stand-in for the taxonomy + rule gloss, which really IS byte-identical across
# every call in a sweep. Repeated to fill whatever shared fraction is requested.
SHARED_BLOCK = (
    "TAXONOMY. STRUCTURING: deposits deliberately kept below the $10,000 CTR "
    "threshold. FUNNEL: many deposits across branches, one outbound wire. "
    "LAYERING: rapid hops through several accounts with no economic purpose. "
    "PASS-THROUGH: in and out within 48h, balance returns to zero. CUCKOO "
    "SMURFING: third-party deposits into an unrelated account. ROUND-TRIPPING. "
    "VELOCITY SPIKE. TRADE-BASED OVER-INVOICING. Benign look-alikes: payroll "
    "companies, franchises depositing to one HQ account, escrow and title "
    "settlement, seasonal businesses, legitimate remittance corridors. "
)


def build_prompt(i, target_chars, shared_frac):
    # type: (int, int, float) -> str
    """Prompt with a byte-identical prefix of `shared_frac`, then unique filler.

    The shared part models the taxonomy + rule that every sweep call really does
    share, so vLLM's prefix cache can legitimately serve it. The rest is unique
    per i, so the cache cannot serve the whole prompt and hand back a number that
    is wrong in the direction you want to believe.
    """
    shared_chars = int(target_chars * shared_frac)
    shared = ""
    if shared_chars:
        reps = shared_chars // len(SHARED_BLOCK) + 1
        shared = (SHARED_BLOCK * reps)[:shared_chars]

    parts = [shared, PREAMBLE]
    j = 0
    while len("".join(parts)) < target_chars:
        ids = ",".join("T%d-%d" % (i, j * 8 + k) for k in range(8))
        parts.append(FILLER.format(
            i=i + j, sector=SECTORS[(i + j) % len(SECTORS)],
            age=(i * 7 + j * 13) % 900 + 30,
            county=COUNTIES[(i + j) % len(COUNTIES)],
            n=(i + j) % 12 + 3, total=((i + j) % 40 + 20) * 2000,
            mx=9000 + ((i + j) % 900), br=(i + j) % 4 + 1,
            st=(i + j) % 3 + 1, ob=(i + j) % 2 + 1, ids=ids))
        j += 1
    return "".join(parts)[:target_chars]


def one_call(url, model, i, prompt_chars, max_tokens, timeout, shared_frac):
    # type: (str, str, int, int, int, int, float) -> Dict
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user",
                      "content": build_prompt(i, prompt_chars, shared_frac)}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        dt = time.time() - t0
        usage = payload.get("usage") or {}
        return {"ok": True, "latency": dt,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0)}
    except Exception as exc:  # noqa: BLE001 - we want every failure mode reported, not raised
        return {"ok": False, "latency": time.time() - t0, "error": repr(exc)[:200]}


def run_level(url, model, conc, n_calls, prompt_chars, max_tokens, timeout,
              offset, shared_frac):
    # type: (str, str, int, int, int, int, int, int, float) -> Dict
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        results = list(ex.map(
            lambda i: one_call(url, model, i, prompt_chars, max_tokens, timeout,
                               shared_frac),
            range(offset, offset + n_calls)))
    wall = time.time() - t0

    ok = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]
    if not ok:
        return {"concurrency": conc, "wall": wall, "ok": 0, "failed": len(bad),
                "first_error": bad[0].get("error") if bad else None}

    lat = sorted(r["latency"] for r in ok)
    out_tok = sum(r["completion_tokens"] for r in ok)
    in_tok = sum(r["prompt_tokens"] for r in ok)
    return {
        "concurrency": conc,
        "wall": wall,
        "ok": len(ok),
        "failed": len(bad),
        "calls_per_sec": len(ok) / wall if wall else 0.0,
        "out_tok_per_sec": out_tok / wall if wall else 0.0,
        "total_tok_per_sec": (out_tok + in_tok) / wall if wall else 0.0,
        "lat_p50": statistics.median(lat),
        "lat_p95": lat[min(len(lat) - 1, int(0.95 * len(lat)))],
        "mean_prompt_tokens": in_tok / len(ok),
        "first_error": bad[0].get("error") if bad else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000/v1/chat/completions",
                    help="inside the OpenShell sandbox this is https://inference.local")
    ap.add_argument("--model", required=True)
    ap.add_argument("--levels", default="1,2,4",
                    help="concurrency levels to test. DO NOT EXCEED 4 on vLLM+NVFP4: "
                         "the MARLIN MoE kernel deadlocks at 8 concurrent and the server "
                         "never recovers (see BOX_STATE.md). Measured 22 Aug on the box.")
    ap.add_argument("--force", action="store_true",
                    help="permit concurrency above 4. Deadlocks vLLM on the NVFP4 MoE path; "
                         "recovery is a container restart. Only on a different engine/model")
    ap.add_argument("--calls", type=int, default=32,
                    help="calls per level; keep constant so levels are comparable")
    ap.add_argument("--prompt-chars", type=int, default=6000,
                    help="~1.5k tokens. Match your real classify payload")
    ap.add_argument("--max-tokens", type=int, default=160,
                    help="match the verdict+rationale you actually expect back")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--shared-prefix", type=float, default=0.65,
                    help="fraction of the prompt that is byte-identical across calls "
                         "(taxonomy+rule). 0.65 matches run_baseline.py. Run 0.0 for the "
                         "no-cache floor; the gap is what prefix caching is worth")
    ap.add_argument("--budget", type=int, default=7200,
                    help="seconds available for the loop (default 2h)")
    ap.add_argument("--out", default="throughput.json")
    args = ap.parse_args()

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    if max(levels) > 4 and not args.force:
        print("REFUSING: concurrency %d requested.\n" % max(levels))
        print("  The NVFP4 MoE MARLIN kernel deadlocks above 4 concurrent on this box.")
        print("  Reproduced 3x: requests stick at Running, 0.0 tok/s, GPU 96% util at 17W.")
        print("  The server never recovers -- it needs a container restart, and on race day")
        print("  that is the loop dying, not an inconvenience. See BOX_STATE.md.\n")
        print("  Use --levels 1,2,4. Pass --force only on a different engine or model.")
        return 1

    print("endpoint : %s" % args.url)
    print("model    : %s" % args.model)
    print("payload  : ~%d chars in, %d tokens out, %d calls per level"
          % (args.prompt_chars, args.max_tokens, args.calls))
    print("prefix   : %.0f%% of each prompt is byte-identical across calls%s\n"
          % (100 * args.shared_prefix,
             "  <- no-cache FLOOR" if args.shared_prefix == 0 else ""))

    print("warming up (one call, ignored) ...", end=" ")
    sys.stdout.flush()
    warm = one_call(args.url, args.model, 999999, args.prompt_chars,
                    args.max_tokens, args.timeout, args.shared_prefix)
    if not warm["ok"]:
        print("FAILED\n\n  %s\n" % warm.get("error"))
        print("The endpoint is not answering. Check the server is up and the URL is right")
        print("(inside the sandbox it is https://inference.local, NOT localhost:8000).")
        return 1
    print("ok, %.1fs\n" % warm["latency"])

    rows = []          # type: List[Dict]
    offset = 0
    hdr = "%5s %8s %8s %10s %11s %9s %9s" % (
        "conc", "wall_s", "ok/fail", "calls/s", "out_tok/s", "p50_s", "p95_s")
    print(hdr)
    print("-" * len(hdr))

    for conc in levels:
        r = run_level(args.url, args.model, conc, args.calls, args.prompt_chars,
                      args.max_tokens, args.timeout, offset, args.shared_prefix)
        offset += args.calls
        rows.append(r)
        if not r.get("calls_per_sec"):
            print("%5d %8.1f %8s   ALL FAILED  %s"
                  % (conc, r["wall"], "0/%d" % r["failed"], r.get("first_error")))
            continue
        print("%5d %8.1f %8s %10.2f %11.1f %9.2f %9.2f" % (
            conc, r["wall"], "%d/%d" % (r["ok"], r["failed"]),
            r["calls_per_sec"], r["out_tok_per_sec"], r["lat_p50"], r["lat_p95"]))

    good = [r for r in rows if r.get("calls_per_sec")]
    if not good:
        print("\nNothing succeeded. Fix the endpoint before reading anything into this.")
        return 1

    best = max(good, key=lambda r: r["calls_per_sec"])
    base = good[0]

    print("\n" + "=" * 62)
    print("BEST CONCURRENCY : %d  (%.2f calls/s, %.1f out tok/s)"
          % (best["concurrency"], best["calls_per_sec"], best["out_tok_per_sec"]))
    if base["concurrency"] == 1:
        print("SPEEDUP vs SERIAL: %.1fx   <- this is what --max-num-seqs buys you"
              % (best["calls_per_sec"] / base["calls_per_sec"]))

    # where does it stop scaling?
    knee = good[0]
    for r in good[1:]:
        if r["calls_per_sec"] < knee["calls_per_sec"] * 1.15:
            break
        knee = r
    if knee["concurrency"] != best["concurrency"]:
        print("KNEE             : %d  (past here you gain <15%% per doubling)"
              % knee["concurrency"])
    if best["concurrency"] == levels[-1] and max(levels) >= 4:
        print("NOTE             : still scaling at 4, but 4 is the HARD CEILING on the")
        print("                   NVFP4 MoE path. Do not raise it. BOX_STATE.md.")
    elif best["concurrency"] == levels[-1]:
        print("NOTE             : still scaling at the top level tested (ceiling is 4).")

    calls = int(best["calls_per_sec"] * args.budget)
    print("\nLOOP SIZING  (budget %ds, at concurrency %d)"
          % (args.budget, best["concurrency"]))
    print("  affordable calls   : ~%d" % calls)
    print("  panel costs 3x     : ~%d windows if every window gets 3 lenses" % (calls // 3))
    print("\n  iterations x dev_set must land under %d:" % calls)
    for dev in (20, 30, 40, 60):
        print("    dev=%-3d ->  %4d iterations" % (dev, calls // dev))
    print("\n  Iterations matter more than dev-set size for a visible curve —")
    print("  but see OPEN_QUESTIONS G1: a 30-window dev set at a 1% base rate holds")
    print("  0.3 positives. Stratify the dev set to ~40% positives.")
    print("=" * 62)

    with open(args.out + ".tmp", "w") as fh:
        json.dump({"args": vars(args), "levels": rows,
                   "best_concurrency": best["concurrency"],
                   "affordable_calls": calls}, fh, indent=2)
    import os
    os.rename(args.out + ".tmp", args.out)   # atomic, per the house rule
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
