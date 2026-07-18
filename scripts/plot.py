#!/usr/bin/env python3
"""Render benchmark plots as dependency-free SVG (Python stdlib only).

Reads a results/*.json (N-engine schema) and writes
results/plots/{latency,throughput,slowdown}.svg. SVG keeps the repo reproducible
(no matplotlib) and renders inline on GitHub.

Usage: python3 scripts/plot.py [results/<file>.json] [outdir]
"""
import glob
import json
import os
import sys

W, H = 760, 380
PAD_L, PAD_R, PAD_T, PAD_B = 70, 20, 40, 60
# Stable colour per engine; keys match the `engines` list in the JSON.
COLORS = {
    "candle-cpu": "#2c7fb8",
    "candle-metal": "#2c7fb8",
    "burn-ndarray": "#de8a26",
    "burn-wgpu": "#de8a26",
    "ort-cpu": "#41ab5d",
    "ort-coreml": "#41ab5d",
}
PALETTE = ["#2c7fb8", "#de8a26", "#41ab5d", "#9e4bb0", "#d6604d"]


def color_for(engine, idx):
    return COLORS.get(engine, PALETTE[idx % len(PALETTE)])


def load():
    path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("results/*.json"))[-1]
    with open(path, encoding="utf-8") as f:
        return json.load(f), path


def svg(body, title):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'font-family="sans-serif" font-size="13">\n'
        f'<rect width="{W}" height="{H}" fill="white"/>\n'
        f'<text x="{W/2}" y="22" text-anchor="middle" font-size="15" '
        f'font-weight="bold">{title}</text>\n{body}</svg>\n'
    )


def axes(ymax, ylabel, xticks):
    px0, px1 = PAD_L, W - PAD_R
    py0, py1 = H - PAD_B, PAD_T
    s = f'<line x1="{px0}" y1="{py0}" x2="{px1}" y2="{py0}" stroke="#333"/>\n'
    s += f'<line x1="{px0}" y1="{py0}" x2="{px0}" y2="{py1}" stroke="#333"/>\n'
    for i in range(6):
        v = ymax * i / 5
        y = py0 - (py0 - py1) * i / 5
        s += f'<line x1="{px0}" y1="{y:.1f}" x2="{px1}" y2="{y:.1f}" stroke="#eee"/>\n'
        s += f'<text x="{px0-8}" y="{y+4:.1f}" text-anchor="end" fill="#555">{v:.0f}</text>\n'
    s += (
        f'<text x="18" y="{(py0+py1)/2:.0f}" text-anchor="middle" fill="#555" '
        f'transform="rotate(-90 18 {(py0+py1)/2:.0f})">{ylabel}</text>\n'
    )
    n = len(xticks)
    for i, t in enumerate(xticks):
        x = px0 + (px1 - px0) * (i + 0.5) / n
        s += f'<text x="{x:.1f}" y="{py0+20}" text-anchor="middle" fill="#555">{t}</text>\n'
    return s, (px0, px1, py0, py1)


def legend(engines, x, y):
    s = ""
    for i, e in enumerate(engines):
        ex = x + i * 130
        s += (
            f'<rect x="{ex}" y="{y}" width="12" height="12" fill="{color_for(e, i)}"/>'
            f'<text x="{ex+16}" y="{y+11}">{e}</text>'
        )
    return s + "\n"


def grouped_bars(records, key_label, engines, mapkey, ymax, ylabel, title):
    xticks = [str(r[key_label]) for r in records]
    body, (px0, px1, py0, py1) = axes(ymax, ylabel, xticks)
    n = len(records)
    e = len(engines)
    slot = (px1 - px0) / n
    bw = slot * 0.8 / e
    for i, r in enumerate(records):
        cx = px0 + slot * (i + 0.5)
        for j, eng in enumerate(engines):
            agg = r[mapkey][eng]
            x = cx - (e * bw) / 2 + j * bw
            h = (py0 - py1) * agg["median"] / ymax
            body += f'<rect x="{x:.1f}" y="{py0-h:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="{color_for(eng, j)}"/>\n'
            elo = (py0 - py1) * agg["p25"] / ymax
            ehi = (py0 - py1) * agg["p75"] / ymax
            xm = x + bw / 2
            body += f'<line x1="{xm:.1f}" y1="{py0-elo:.1f}" x2="{xm:.1f}" y2="{py0-ehi:.1f}" stroke="#333"/>\n'
    body += legend(engines, px0, PAD_T - 4)
    return svg(body, title)


def line_chart(records, xkey, engines, mapkey, ymax, ylabel, title):
    xticks = [f'b={r[xkey]}' for r in records]
    body, (px0, px1, py0, py1) = axes(ymax, ylabel, xticks)
    n = len(records)
    for j, eng in enumerate(engines):
        pts = []
        for i, r in enumerate(records):
            x = px0 + (px1 - px0) * (i + 0.5) / n
            y = py0 - (py0 - py1) * r[mapkey][eng]["median"] / ymax
            pts.append(f"{x:.1f},{y:.1f}")
            body += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color_for(eng, j)}"/>\n'
        body += f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color_for(eng, j)}" stroke-width="2"/>\n'
    body += legend(engines, px0, PAD_T - 4)
    return svg(body, title)


def slowdown_bars(latency, throughput, engines, title):
    """Each engine's slowdown vs the per-scenario fastest (fastest = 1.0x)."""
    rows = []
    for r in latency:
        best = min(r["ms"][e]["median"] for e in engines)
        rows.append((r["seq_label"], {e: r["ms"][e]["median"] / best for e in engines}))
    for r in throughput:
        best = max(r["sps"][e]["median"] for e in engines)
        rows.append((f'thr b={r["batch"]}', {e: best / r["sps"][e]["median"] for e in engines}))
    ymax = max(2.0, max(v for _, d in rows for v in d.values()) * 1.1)
    xticks = [k for k, _ in rows]
    body, (px0, px1, py0, py1) = axes(ymax, "slowdown vs fastest (x)", xticks)
    y1 = py0 - (py0 - py1) * 1.0 / ymax
    body += f'<line x1="{px0}" y1="{y1:.1f}" x2="{px1}" y2="{y1:.1f}" stroke="#c00" stroke-dasharray="4"/>\n'
    body += f'<text x="{px1-2}" y="{y1-4:.1f}" text-anchor="end" fill="#c00">1.0 (fastest)</text>\n'
    n = len(rows)
    e = len(engines)
    slot = (px1 - px0) / n
    bw = slot * 0.8 / e
    for i, (_, d) in enumerate(rows):
        cx = px0 + slot * (i + 0.5)
        for j, eng in enumerate(engines):
            v = d[eng]
            x = cx - (e * bw) / 2 + j * bw
            h = (py0 - py1) * v / ymax
            body += f'<rect x="{x:.1f}" y="{py0-h:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="{color_for(eng, j)}"/>\n'
    body += legend(engines, px0, PAD_T - 4)
    return svg(body, title)


def main():
    data, path = load()
    engines = data["engines"]
    lat = data["latency"]
    thr = data["throughput"]
    cpu = data["environment"]["cpu"]
    sub = f"{cpu}, 1 thread, AC — {data['config']['trials']} trials (median, IQR whiskers)"

    lat_ymax = max(r["ms"][e]["p75"] for r in lat for e in engines) * 1.15
    thr_ymax = max(r["sps"][e]["median"] for r in thr for e in engines) * 1.2

    outdir = sys.argv[2] if len(sys.argv) > 2 else "results/plots"
    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/latency.svg", "w", encoding="utf-8") as f:
        f.write(
            grouped_bars(lat, "seq_label", engines, "ms", lat_ymax,
                         "p50 latency (ms)", f"Latency (lower is better) — {sub}")
        )
    with open(f"{outdir}/throughput.svg", "w", encoding="utf-8") as f:
        f.write(
            line_chart(thr, "batch", engines, "sps", thr_ymax,
                       "sentences / sec", f"Throughput (higher is better) — {sub}")
        )
    with open(f"{outdir}/slowdown.svg", "w", encoding="utf-8") as f:
        f.write(
            slowdown_bars(lat, thr, engines, f"Slowdown vs fastest engine — {sub}")
        )
    print(f"plotted from {path} -> {outdir}/{{latency,throughput,slowdown}}.svg")


if __name__ == "__main__":
    main()
