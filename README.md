```markdown
# AML Red/Blue Adversarial Rule Engine

A synthetic-data engine that stress-tests anti-money-laundering (AML) transaction-monitoring
rules using an adversarial red/blue loop — measuring how a rule set performs against realistic,
Census-anchored transaction data before it ever reaches production.

## The problem

Bank AML rule sets are notoriously imprecise. Published industry false-positive rates run
95–99%. Most teams have no repeatable way to measure a candidate rule against a labeled,
realistic corpus before shipping it — so rules get tuned against production complaints instead
of ground truth.

## What this does

1. **Generates a synthetic transaction corpus** — accounts, transactions, and 14-day
   monitoring windows with exact ground-truth labels — anchored to real US Census County
   Business Patterns (CBP) 2022 data for sector mix and cash intensity.
2. **Extracts a deterministic fact vocabulary** from each window (no model calls — pure
   feature computation).
3. **Runs candidate rules** against that fact vocabulary and scores precision, recall, F1,
   specificity, and false-positive rate — broken out by typology.
4. **Adversarially rewrites rules** (blue) against a fixed rule-evaluation oracle to improve
   F1 without overfitting to a leaked signal.

## Measured results

Corpus: 600 accounts · 84,929 transactions · 7,200 fourteen-day windows · 72 planted
typologies (six typologies × 12 each, 1% base rate). Fully reproducible, stdlib-only,
Python 3.9, runs in ~8 seconds on a laptop.

| Rule | Alerts | Precision | Recall | F1 | Specificity | FP rate |
|---|---|---|---|---|---|---|
| Literal $10,000 CTR threshold | 512 | 0.043 | 0.306 | 0.075 | 0.931 | 95.7% |
| Bank scenario set (4 scenarios) | 1,937 | 0.033 | 0.875 | 0.063 | 0.737 | 96.7% |
| Seed rule (adversarial starting point) | 530 | 0.034 | 0.250 | 0.060 | 0.928 | 96.6% |

Stable across seeds (7 / 11 / 23 / 42): FP rate 96.7–97.0%, F1 0.058–0.063 — this reproduces
the industry-cited 95–99% false-positive regime from a 1% base rate, not by tuning toward it.

**Key finding:** a literal transaction-threshold rule (e.g. flag anything over $10k) is a
weak comparator — it misses 100% of layering cases and 75%+ of structuring, cuckoo-smurfing,
and pass-through cases in this corpus, because those typologies are specifically designed to
stay under the threshold. A realistic multi-scenario rule set is the correct baseline to
measure against, not the single-threshold rule.

## Architecture

```
corpus.py → facts.py → rules.py (fires()) → score.py
                              ↑
                        red.py / blue.py (adversarial rewrite loop)
```

- `corpus.py` — synthetic account/transaction/window generation, Census-anchored
- `facts.py` — deterministic fact extraction per window
- `rules.py` — rule schema, evaluator, and baseline rule sets
- `score.py` — precision/recall/F1/specificity/FP-rate, by typology
- `red.py` / `blue.py` — adversarial rewrite loop
- `dashboard.py` / `chart.py` / `demo_live.py` — live visualization of the loop
- `run_baseline.py` — end-to-end driver

## Running it

```bash
python3 build/run_baseline.py --accounts 600 --base-rate 0.01 --seed 7
```

No dependencies — pure Python 3.9 standard library. Full output written to
`build/out/baseline_report.txt`.

## Tech stack

Python 3.9 (stdlib only) · adversarial rule-rewrite loop backed by an LLM (see `red.py`/`blue.py`)
· US Census CBP 2022 as the real-world data anchor.

## Background

Built at the Dell × NVIDIA Hackathon. This repository is a cleaned, public version of the
original hackathon project, containing the implementation and measured results only.
