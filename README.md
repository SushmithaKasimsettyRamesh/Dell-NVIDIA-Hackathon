# AML Red/Blue Adversarial Rule Engine

A synthetic-data engine for stress-testing anti-money-laundering (AML) transaction-monitoring rules using an adversarial red/blue loop and Census-anchored transaction data.

## The Problem

AML transaction-monitoring rules can generate large numbers of false positives. Yet teams often lack a repeatable way to evaluate candidate rules against labeled, realistic data before deploying them.

## What It Does

1. **Generates synthetic transaction data** — accounts, transactions, and 14-day monitoring windows with ground-truth labels, anchored to US Census County Business Patterns (CBP) 2022 data.
2. **Extracts deterministic features** from each monitoring window without model calls.
3. **Evaluates candidate rules** using precision, recall, F1, specificity, and false-positive rate, including breakdowns by typology.
4. **Adversarially rewrites rules** through a red/blue loop against a fixed evaluation oracle to improve F1 while guarding against leaked signals.

## Measured Results

**Corpus:** 600 accounts · 84,929 transactions · 7,200 fourteen-day windows · 72 planted typologies · 1% base rate

Fully reproducible, Python 3.9 standard library only, with a runtime of approximately 8 seconds on a laptop.

| Rule                            | Alerts | Precision | Recall |    F1 | Specificity | FP Rate |
| ------------------------------- | -----: | --------: | -----: | ----: | ----------: | ------: |
| Literal $10,000 CTR threshold   |    512 |     0.043 |  0.306 | 0.075 |       0.931 |   95.7% |
| Bank scenario set (4 scenarios) |  1,937 |     0.033 |  0.875 | 0.063 |       0.737 |   96.7% |
| Seed rule                       |    530 |     0.034 |  0.250 | 0.060 |       0.928 |   96.6% |

Across seeds 7, 11, 23, and 42, the bank scenario baseline remained in a similar range: **96.7–97.0% false-positive rate and 0.058–0.063 F1**.

### Key Finding

A literal $10,000 transaction threshold is a weak comparator for several typologies. In this corpus, it misses 100% of layering cases and 75%+ of structuring, cuckoo-smurfing, and pass-through cases because these behaviors are designed to remain below simple transaction thresholds.

A multi-scenario rule set provides a more meaningful baseline for evaluating AML detection performance.

## Architecture

```text
corpus.py → facts.py → rules.py → score.py
                 ↑
          red.py / blue.py
       adversarial rewrite loop
```

### Core Components

* `corpus.py` — synthetic account, transaction, and monitoring-window generation
* `facts.py` — deterministic feature extraction
* `rules.py` — rule schema, evaluator, and baseline rule sets
* `score.py` — precision, recall, F1, specificity, and false-positive rate
* `red.py` / `blue.py` — adversarial rule-rewrite loop
* `dashboard.py` / `chart.py` / `demo_live.py` — visualization and live demonstration
* `run_baseline.py` — end-to-end benchmark driver

## Running It

```bash
python3 build/run_baseline.py --accounts 600 --base-rate 0.01 --seed 7
```

No external dependencies — Python 3.9 standard library only.

Output is written to:

```text
build/out/baseline_report.txt
```

## Tech Stack

**Python 3.9** · Standard Library · Adversarial Rule Rewriting · Synthetic Data · AML Transaction Monitoring · US Census CBP 2022

## Background

Built at the **Dell × NVIDIA Hackathon** and cleaned for public release. This repository contains the implementation and measured results from the AML rule-testing engine.

