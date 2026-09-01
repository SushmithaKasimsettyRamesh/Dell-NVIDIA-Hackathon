# build/ — the engine

Stdlib-only, Python 3.9. No third-party imports, nothing to install, runs on the laptop
and on the box unchanged.

```
python3 build/run_baseline.py [--accounts 600] [--base-rate 0.01] [--seed 7]
```

| File | What |
|---|---|
| `corpus.py` | synthetic accounts / transactions / 14-day windows / exact labels. Census-anchored via `data/business_mix.json` |
| `facts.py` | the fact vocabulary from `RED_BLUE_SPEC.md` §4. Deterministic. Every fact emitted for every window |
| `rules.py` | rule schema + `fires()` evaluator (§3), plus three baselines: literal CTR, bank scenario set, seed rule |
| `score.py` | precision / recall / F1 / specificity / FP-rate, and recall broken out by typology |
| `run_baseline.py` | driver: builds the corpus, scores all three rules, and sizes the day's model workload |

Writes `out/{accounts.json,txns.jsonl,labels.jsonl,facts.jsonl,baseline_results.json,baseline_report.txt}`.

**Results and what they mean: `../BASELINE_RESULT.md`.**

🔴 The four throughput assumptions at the top of `run_baseline.py` are guesses. Measure one
call on the box at 11:00, set them, rerun. Every wall clock scales linearly with them.

## What's here now
| File | |
|---|---|
| `measure_throughput.py` | ⛳️ **Run first, the moment the endpoint answers.** Sweeps concurrency levels, reports calls/s, finds the knee, prints affordable `iterations x dev_set` against your time budget. Stdlib, Python 3.9, ~40 s |
| `openshell-policy.yaml` | Starter deny-by-default sandbox policy |
| `data/business_mix.json` | US Census CBP 2022 — sector mix + Cincinnati tri-state footprint |

```bash
python3 build/measure_throughput.py --model nvidia/Qwen3.6-35B-A3B-NVFP4 --budget 7200
python3 build/measure_throughput.py --model nvidia/Qwen3.6-35B-A3B-NVFP4 --shared-prefix 0.0
python3 build/measure_throughput.py --url http://localhost:8081/v1/chat/completions \
    --model nemotron --levels 1,2,4          # llama.cpp
```
Run it at `--shared-prefix 0.65` (realistic — taxonomy+rule are identical across sweep calls) and
`0.0` (floor). **Then overwrite `PREFILL_TOK_PER_S`, `DECODE_TOK_PER_S` and `SHARED_PREFIX_FRAC` at
the top of `run_baseline.py`** — they are currently estimates, and all of `BASELINE_RESULT.md` §4
depends on them.

🔴 **Everything that calls the endpoint must use `ThreadPoolExecutor`, never a serial `for` loop** —
the server's `--max-num-seqs` does nothing without concurrent clients. `ENGINE.md` §3.
