"""Metrics. Precision / recall / F1 / specificity / FP rate, plus alert volume.

FP rate here is the AML sense: share of ALERTS that are not true positives — the
number the 95% figure refers to. Not 1 - specificity.
"""
def confusion(pred, labels):
    tp = fp = fn = tn = 0
    for wid, p in pred.items():
        y = labels.get(wid) is not None
        if p and y:
            tp += 1
        elif p and not y:
            fp += 1
        elif (not p) and y:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def metrics(pred, labels):
    c = confusion(pred, labels)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    prec = tp / float(tp + fp) if (tp + fp) else 0.0
    rec = tp / float(tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    spec = tn / float(tn + fp) if (tn + fp) else 0.0
    c.update({"precision": prec, "recall": rec, "f1": f1, "specificity": spec,
              "fp_rate": 1.0 - prec if (tp + fp) else 0.0,
              "alerts": tp + fp, "positives": tp + fn, "windows": tp + fp + fn + tn})
    return c


def objective(m, kind="f2", n_windows=None, budget=0.15):
    """What blue optimises. NOT F1.

    F1 drives blue to a sniper rule -- measured: precision 1.000, recall 0.164, eleven
    alerts. That breaks the two-stage design: L0 is meant to be a high-RECALL gate that
    the panel then makes precise. If L0 is already perfectly precise the panel has no
    work, and system recall caps at 0.16. Banks do not ship that; the loss function is
    deliberately asymmetric (FALSE_POSITIVES.md reason 3).

      f1      balanced -- kept only for comparison
      f2      recall weighted 2x, the standard asymmetric-loss choice
      budget  maximise recall subject to an alert budget, which is how a bank actually
              tunes: "we can review N alerts a month, catch as much as you can inside it"
    """
    if kind == "budget":
        cap = budget * n_windows
        if m["alerts"] <= cap:
            return m["recall"]
        return m["recall"] * (cap / float(m["alerts"]))     # graded, not a cliff
    beta = 2.0 if kind == "f2" else 1.0
    p, r = m["precision"], m["recall"]
    d = (beta * beta * p) + r
    return (1 + beta * beta) * p * r / d if d else 0.0


def fmt(name, m):
    return ("%-34s alerts %5d | TP %3d FP %5d FN %3d | P %.3f  R %.3f  F1 %.3f | "
            "spec %.4f | FP-rate %.1f%%" %
            (name[:34], m["alerts"], m["tp"], m["fp"], m["fn"],
             m["precision"], m["recall"], m["f1"], m["specificity"], 100.0 * m["fp_rate"]))


def recall_by_typology(pred, labels):
    """Which typologies a rule actually catches. An overall recall number hides
    a rule that catches only the easy family."""
    out = {}
    for wid, typ in labels.items():
        if typ is None:
            continue
        got, tot = out.get(typ, (0, 0))
        out[typ] = (got + (1 if pred.get(wid) else 0), tot + 1)
    return out


# ---------------------------------------------------------------- cascade metrics
# The cascade can only remove. Precision rises because alerts get closed, and some
# closed alerts are real -- so recall must be reported at every stage, not just at
# the end. See RUN_PLAN.md section 8.

def cascade_metrics(labels, l0_alerts, qwen_flag, nemo_flag, n_planted):
    """l0_alerts: [window_id] the rule fired on.
       qwen_flag: {window_id: bool} intent verdict, all alerts.
       nemo_flag: {window_id: bool} literal verdict, subset (flags + audit sample).
       Returns stage-by-stage recall so the precision/recall trade is visible."""
    truth = lambda w: labels.get(w) is not None
    l0_tp = [w for w in l0_alerts if truth(w)]

    kept = [w for w in l0_alerts if qwen_flag.get(w)]
    kept_tp = [w for w in kept if truth(w)]
    cleared_tp = [w for w in l0_alerts if truth(w) and not qwen_flag.get(w)]

    auto, esc_split, esc_audit = [], [], []
    for w in l0_alerts:
        q, n = qwen_flag.get(w), nemo_flag.get(w)
        if q and n is True:
            auto.append(w)
        elif q and n is False:
            esc_split.append(w)
        elif (not q) and n is True:
            esc_audit.append(w)          # Qwen missed it, the audit caught it
    reviewed = auto + esc_split + esc_audit

    def prec(ws):
        return len([w for w in ws if truth(w)]) / float(len(ws)) if ws else 0.0

    return {
        "planted": n_planted,
        "l0_alerts": len(l0_alerts), "l0_tp": len(l0_tp),
        "l0_precision": prec(l0_alerts),
        # 🔴 SCOPE. l0_alerts is usually a SAMPLE of the alerts, so dividing its true
        # positives by all planted understates recall by exactly the sampling fraction.
        # It printed "L0 recall 0.292" when the real figure was 0.667. Named to make the
        # scope impossible to misread, and the full-corpus recall must be passed in.
        "recall_within_sample": len(l0_tp) / float(n_planted) if n_planted else 0.0,
        "panel_retention": None,
        "l0_missed": n_planted - len(l0_tp),

        "qwen_kept": len(kept), "qwen_kept_tp": len(kept_tp),
        "qwen_precision": prec(kept),
        "qwen_cleared": len(l0_alerts) - len(kept),
        "qwen_cleared_tp": len(cleared_tp),      # the silent loss -- name it on stage
        # panel_retention x full-corpus L0 recall = true end-to-end system recall
        "panel_retention_of_l0": len(kept_tp) / float(len(l0_tp)) if l0_tp else 0.0,

        "auto_file": len(auto), "auto_file_precision": prec(auto),
        "escalated_split": len(esc_split), "escalated_audit": len(esc_audit),
        "escalation_rate": len(esc_split + esc_audit) / float(len(l0_alerts)) if l0_alerts else 0.0,
        "reviewed_precision": prec(reviewed),
        "audit_recovered_tp": len([w for w in esc_audit if truth(w)]),
    }


def audit_sample(labels, l0_alerts, qwen_flag, n_negatives, rnd):
    """STRATIFIED, using ground truth -- a uniform sample cannot see a rare miss.
    ~5 true positives hide in ~1,000 cleared alerts; a uniform 150-sample finds
    ZERO 44% of the time and would report a miss rate of 0 as a measurement artifact.

    This is a MEASUREMENT instrument, not part of the detection path -- no verdict
    anywhere in the system sees a label. Say that out loud on stage."""
    cleared = [w for w in l0_alerts if not qwen_flag.get(w)]
    tp = [w for w in cleared if labels.get(w) is not None]        # every one of them
    fp = [w for w in cleared if labels.get(w) is None]
    rnd.shuffle(fp)
    return tp + fp[:n_negatives], len(tp), len(cleared)
