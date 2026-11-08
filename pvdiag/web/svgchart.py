"""服务端 SVG 图表（无任何外部 JS/CDN 依赖，内网/私有化部署可用）"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence
from xml.sax.saxutils import escape

PALETTE = ["#b91c1c", "#111827", "#2563eb", "#059669", "#d97706", "#7c3aed", "#db2777", "#0891b2", "#65a30d", "#9333ea", "#ea580c", "#4b5563"]


def _nice_ticks(lo: float, hi: float, n: int = 5) -> List[float]:
    if hi <= lo:
        hi = lo + 1
    span = hi - lo
    raw = span / n
    mag = 10 ** int(f"{raw:e}".split("e")[1])
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=mag)
    start = int(lo // step) * step
    out, t = [], start
    while t <= hi + 1e-9:
        if t >= lo - 1e-9:
            out.append(round(t, 6))
        t += step
    return out


def line_chart(labels: Sequence[str], series: List[Dict], title: str = "", y_label: str = "",
               width: int = 620, height: int = 260, label_every: int = 4) -> str:
    """series: [{label, data: [float|None,...]}]；第 1 条加粗红色，第 2 条黑色虚线（群组中位）"""
    ml, mr, mt, mb = 46, 12, 28, 34
    W, H = width - ml - mr, height - mt - mb
    vals = [v for s in series for v in s["data"] if v is not None]
    lo, hi = (0.0, 1.0) if not vals else (min(0.0, min(vals)), max(vals) * 1.08 or 1.0)
    ticks = _nice_ticks(lo, hi)
    hi = max(hi, ticks[-1]) if ticks else hi
    n = max(len(labels), 2)
    sx = lambda i: ml + i * W / (n - 1)
    sy = lambda v: mt + H - (v - lo) / (hi - lo) * H
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" style="max-height:{height}px;font:11px sans-serif">']
    if title:
        parts.append(f'<text x="{ml}" y="16" font-size="12" font-weight="600" fill="#1d2433">{escape(title)}</text>')
    for t in ticks:
        y = sy(t)
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + W}" y2="{y:.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{ml - 6}" y="{y + 4:.1f}" text-anchor="end" fill="#6b7280">{t:g}</text>')
    for i, lab in enumerate(labels):
        if i % label_every == 0:
            parts.append(f'<text x="{sx(i):.1f}" y="{mt + H + 14}" text-anchor="middle" fill="#6b7280">{escape(str(lab))}</text>')
    if y_label:
        parts.append(f'<text x="{ml - 6}" y="{mt - 8}" text-anchor="end" fill="#6b7280">{escape(y_label)}</text>')
    parts.append(f'<line x1="{ml}" y1="{mt + H}" x2="{ml + W}" y2="{mt + H}" stroke="#9ca3af"/>')
    many = len(series) > 2
    for k, s in enumerate(series):
        color = PALETTE[k % len(PALETTE)] if many else ("#b91c1c" if k == 0 else "#111827")
        width_px = 2.5 if (k == 0 and not many) else 1.4
        dash = ' stroke-dasharray="6 4"' if (k == 1 and not many) else ""
        seg, d = [], ""
        for i, v in enumerate(s["data"]):
            if v is None:
                if d:
                    seg.append(d)
                d = ""
                continue
            d += f'{"M" if not d else "L"}{sx(i):.1f} {sy(v):.1f} '
        if d:
            seg.append(d)
        for path in seg:
            parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{width_px}"{dash}/>')
    # 图例
    x = ml
    for k, s in enumerate(series[:8]):
        color = PALETTE[k % len(PALETTE)] if many else ("#b91c1c" if k == 0 else "#111827")
        parts.append(f'<rect x="{x}" y="{height - 12}" width="14" height="3" fill="{color}"/>')
        parts.append(f'<text x="{x + 18}" y="{height - 8}" fill="#374151">{escape(str(s["label"]))}</text>')
        x += 18 + 7 * len(str(s["label"])) + 14
    parts.append("</svg>")
    return "".join(parts)


def bar_chart(labels: Sequence[str], values: Sequence[Optional[float]], colors: Sequence[str], title: str = "",
              hlines: Sequence[float] = (-3.0, -5.0), width: int = 620, height: int = 260, y_min: float = -15, y_max: float = 5) -> str:
    ml, mr, mt, mb = 40, 12, 28, 40
    W, H = width - ml - mr, height - mt - mb
    vals = [v for v in values if v is not None]
    lo = min(y_min, min(vals) if vals else y_min)
    hi = max(y_max, max(vals) if vals else y_max)
    lo = max(lo, -100.0)
    sy = lambda v: mt + H - (max(v, lo) - lo) / (hi - lo) * H
    n = max(len(labels), 1)
    bw = max(2.0, W / n * 0.7)
    sx = lambda i: ml + (i + 0.5) * W / n
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" style="max-height:{height}px;font:11px sans-serif">']
    if title:
        parts.append(f'<text x="{ml}" y="16" font-size="12" font-weight="600" fill="#1d2433">{escape(title)}</text>')
    for t in _nice_ticks(lo, hi, 6):
        y = sy(t)
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + W}" y2="{y:.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{ml - 6}" y="{y + 4:.1f}" text-anchor="end" fill="#6b7280">{t:g}%</text>')
    for hl in hlines:
        if lo <= hl <= hi:
            y = sy(hl)
            parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + W}" y2="{y:.1f}" stroke="#b91c1c" stroke-dasharray="4 3"/>')
            parts.append(f'<text x="{ml + W}" y="{y - 3:.1f}" text-anchor="end" fill="#b91c1c">{hl:g}%</text>')
    y0 = sy(0)
    parts.append(f'<line x1="{ml}" y1="{y0:.1f}" x2="{ml + W}" y2="{y0:.1f}" stroke="#9ca3af"/>')
    for i, (v, c) in enumerate(zip(values, colors)):
        if v is None:
            continue
        y = sy(v)
        top, h = (min(y, y0), abs(y - y0))
        parts.append(f'<rect x="{sx(i) - bw / 2:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{max(h, 1):.1f}" fill="{c}"><title>{escape(str(labels[i]))}: {v:+.1f}%</title></rect>')
    step = max(1, n // 8)
    for i, lab in enumerate(labels):
        if i % step == 0:
            parts.append(f'<text x="{sx(i):.1f}" y="{mt + H + 14}" text-anchor="middle" fill="#6b7280">{escape(str(lab)[5:])}</text>')
    legend = [("#0f766e", "晴好天"), ("#f59e0b", "多云"), ("#9ca3af", "阴雨/无数据")]
    x = ml
    for c, t in legend:
        parts.append(f'<rect x="{x}" y="{height - 12}" width="12" height="8" fill="{c}"/><text x="{x + 16}" y="{height - 4}" fill="#374151">{t}</text>')
        x += 16 + 7 * len(t) + 16
    parts.append("</svg>")
    return "".join(parts)
