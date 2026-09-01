"""The curve, as inline SVG. Stdlib only -- no matplotlib on the box.

    python3 build/chart.py            -> out/curve.svg

Plots the accepted path AND every rejected candidate. The loop keeps only
improvements, so the precision/recall trade-off is invisible on the accepted line;
the scatter is where "it caught more, and flagged every payroll company doing it"
is actually visible. That is the beat at PITCH.md 2:00.
"""
import json, os, sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
# separate margins: the end labels live in a right gutter. With a single PAD the
# labels started at x(n-1)+8 and ran past the viewBox, clipping "precision 0.055".
W, H = 980, 400
ML, MR, MT, MB = 62, 132, 58, 46
INK, GRID = "#1a1a1a", "#e4e4e4"
REC, PRE, REJ = "#c0392b", "#2471a3", "#b8b8b8"


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def load():
    rows = [json.loads(l) for l in open(os.path.join(OUT, "loop.jsonl"))]
    it = [r for r in rows if r.get("event") == "iteration" and r.get("seconds", 999) < 40]
    cand = [r for r in rows if r.get("event") == "candidate"]
    return it, cand, [r for r in rows if r.get("event") == "start"]


def svg_curve(it, cand, start):
    n = max(1, len(it))
    x = lambda i: ML + (W - ML - MR) * (i / float(max(1, n - 1)))
    y = lambda v: H - MB - (H - MT - MB) * min(1.0, v / 0.75)
    p = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="100%%" '
         'font-family="Georgia,serif">' % (W, H),
         '<rect width="%d" height="%d" fill="#fdfdfb"/>' % (W, H)]
    for g in (0.0, 0.15, 0.30, 0.45, 0.60, 0.75):
        p.append('<line x1="%d" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s"/>'
                 % (ML, y(g), W - MR, y(g), GRID))
        p.append('<text x="%d" y="%.1f" font-size="11" fill="#777" text-anchor="end">%.2f</text>'
                 % (ML - 9, y(g) + 4, g))
    # iteration ticks
    step = max(1, n // 6)
    for i in range(0, n, step):
        p.append('<text x="%.1f" y="%d" font-size="10.5" fill="#999" text-anchor="middle">%d'
                 '</text>' % (x(i), H - MB + 17, it[i]["i"]))
    p.append('<line x1="%d" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#c9c9c9"/>'
             % (ML, y(0), W - MR, y(0)))

    for c in cand:
        i = min(n - 1, c["i"] - 1)
        p.append('<circle cx="%.1f" cy="%.1f" r="2.4" fill="%s" opacity="0.5"/>'
                 % (x(i), y(c["recall"]), REJ))

    # keep the two end labels from colliding when the lines finish close together
    ys = {}
    for key, col, lab in (("recall", REC, "recall"), ("precision", PRE, "precision")):
        pts = " ".join("%.1f,%.1f" % (x(k), y(r[key])) for k, r in enumerate(it))
        p.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="2.6"/>'
                 % (pts, col))
        ys[key] = (y(it[-1][key]), col, lab, it[-1][key])
    a, b = ys["recall"], ys["precision"]
    if abs(a[0] - b[0]) < 18:
        lo, hi = (a, b) if a[0] < b[0] else (b, a)
        ys = {lo[2]: (lo[0] - 9,) + lo[1:], hi[2]: (hi[0] + 9,) + hi[1:]}
    for yy, col, lab, val in ys.values():
        p.append('<circle cx="%.1f" cy="%.1f" r="3.4" fill="%s"/>'
                 % (x(n - 1), y(val), col))
        p.append('<text x="%.1f" y="%.1f" font-size="13" fill="%s">%s %.3f</text>'
                 % (W - MR + 12, yy + 4, col, lab, val))

    s = start[-1] if start else {}
    p.append('<text x="%d" y="27" font-size="15.5" fill="%s">Detection rule, rewritten by '
             'the model %d times</text>' % (ML, INK, n))
    p.append('<text x="%d" y="46" font-size="11.5" fill="#666">grey dots are candidates it '
             'tried and rejected &#183; dev set %s windows, %s planted &#183; scored in '
             'Python, 0 model calls</text>'
             % (ML, format(s.get("dev", 0), ",d") if s.get("dev") else "?",
                s.get("dev_positives", "?")))
    p.append('<text x="%.1f" y="%d" font-size="11.5" fill="#888" text-anchor="middle">'
             'iteration</text>' % ((ML + W - MR) / 2.0, H - 12))
    p.append("</svg>")
    return "\n".join(p)


def main():
    it, cand, start = load()
    if not it:
        print("no clean iterations in the ledger")
        return
    svg = svg_curve(it, cand, start)
    open(os.path.join(OUT, "curve.svg"), "w").write(svg)
    a, b = it[0], it[-1]
    print("curve.svg: %d iterations, %d candidates" % (len(it), len(cand)))
    print("  recall    %.3f -> %.3f" % (a["recall"], b["recall"]))
    print("  precision %.3f -> %.3f" % (a["precision"], b["precision"]))
    print("  alerts    %d -> %d" % (a["alerts"], b["alerts"]))
    h = [r for r in it if r.get("holdout_f1") is not None]
    if h:
        print("  holdout F1 checkpoints: %s" % ", ".join("%.3f" % r["holdout_f1"] for r in h))


if __name__ == "__main__":
    main()
