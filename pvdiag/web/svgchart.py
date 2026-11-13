"""服务端 SVG 图表（无任何外部 JS/CDN 依赖，内网/私有化部署可用）。
颜色全部走 CSS 变量（--c-*），随页面明暗主题切换。"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence
from xml.sax.saxutils import escape

SERIES = ["var(--c-fault)", "var(--c-ink)", "var(--c-sky)", "var(--c-ok)", "var(--c-low)", "var(--c-orient)",
          "#c2528a", "#1d8f9c", "#6a9a1f", "#8a4fd6", "#d8722a", "var(--c-muted)"]


def _nice_ticks(lo: float, hi: float, n: int = 5) -> List[float]:
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / n
    mag = 10 ** int(f"{raw:e}".split("e")[1])
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=mag)
    out, t = [], int(lo // step) * step
    while t <= hi + 1e-9:
        if t >= lo - 1e-9:
            out.append(round(t, 6))
        t += step
    return out


def line_chart(labels: Sequence[str], series: List[Dict], title: str = "", y_label: str = "",
               width: int = 640, height: int = 260, label_every: int = 4) -> str:
    """series: [{label, data: [float|None,...]}]；两条线时第 1 条为该串（粗、故障色）、第 2 条为群组中位（虚线）"""
    ml, mr, mt, mb = 44, 12, 26, 36
    W, H = width - ml - mr, height - mt - mb
    vals = [v for s in series for v in s["data"] if v is not None]
    lo, hi = (0.0, 1.0) if not vals else (min(0.0, min(vals)), (max(vals) or 1.0) * 1.08)
    ticks = _nice_ticks(lo, hi)
    hi = max(hi, ticks[-1]) if ticks else hi
    n = max(len(labels), 2)
    sx = lambda i: ml + i * W / (n - 1)
    sy = lambda v: mt + H - (v - lo) / (hi - lo) * H
    p = [f'<svg class="chart" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{escape(title)}">']
    if title:
        p.append(f'<text x="{ml}" y="15" class="t">{escape(title)}</text>')
    for t in ticks:
        y = sy(t)
        p.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + W}" y2="{y:.1f}" class="grid"/>')
        p.append(f'<text x="{ml - 6}" y="{y + 4:.1f}" text-anchor="end" class="ax">{t:g}</text>')
    for i, lab in enumerate(labels):
        if i % label_every == 0:
            p.append(f'<text x="{sx(i):.1f}" y="{mt + H + 14}" text-anchor="middle" class="ax">{escape(str(lab))}</text>')
    if y_label:
        p.append(f'<text x="{ml - 6}" y="{mt - 8}" text-anchor="end" class="ax">{escape(y_label)}</text>')
    p.append(f'<line x1="{ml}" y1="{mt + H}" x2="{ml + W}" y2="{mt + H}" class="axis"/>')
    many = len(series) > 2
    for k, s in enumerate(series):
        color = SERIES[k % len(SERIES)] if many else ("var(--c-fault)" if k == 0 else "var(--c-ink)")
        sw = 2.4 if (k == 0 and not many) else 1.4
        dash = ' stroke-dasharray="6 4"' if (k == 1 and not many) else ""
        segs, d = [], ""
        for i, v in enumerate(s["data"]):
            if v is None:
                if d:
                    segs.append(d)
                d = ""
                continue
            d += f'{"M" if not d else "L"}{sx(i):.1f} {sy(v):.1f} '
        if d:
            segs.append(d)
        for path in segs:
            p.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linejoin="round"{dash}/>')
    x = ml
    for k, s in enumerate(series[:8]):
        color = SERIES[k % len(SERIES)] if many else ("var(--c-fault)" if k == 0 else "var(--c-ink)")
        p.append(f'<rect x="{x}" y="{height - 11}" width="14" height="3" fill="{color}"/>')
        p.append(f'<text x="{x + 18}" y="{height - 7}" class="lg">{escape(str(s["label"]))}</text>')
        x += 18 + 7 * len(str(s["label"])) + 14
    p.append("</svg>")
    return "".join(p)


def bar_chart(labels: Sequence[str], values: Sequence[Optional[float]], weather: Sequence[str], title: str = "",
              hlines: Sequence[float] = (-3.0, -5.0), width: int = 640, height: int = 260, y_min: float = -15, y_max: float = 5) -> str:
    """偏差趋势：按天一根柱，颜色 = 天气（晴好/多云/阴雨），红虚线 = 阈值"""
    ml, mr, mt, mb = 40, 12, 26, 40
    W, H = width - ml - mr, height - mt - mb
    vals = [v for v in values if v is not None]
    lo = max(min(y_min, min(vals) if vals else y_min), -100.0)
    hi = max(y_max, max(vals) if vals else y_max)
    sy = lambda v: mt + H - (max(v, lo) - lo) / (hi - lo) * H
    n = max(len(labels), 1)
    bw = max(2.0, W / n * 0.68)
    sx = lambda i: ml + (i + 0.5) * W / n
    wcolor = {"clear": "var(--c-clear)", "cloudy": "var(--c-cloudy)"}
    p = [f'<svg class="chart" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{escape(title)}">']
    if title:
        p.append(f'<text x="{ml}" y="15" class="t">{escape(title)}</text>')
    for t in _nice_ticks(lo, hi, 6):
        y = sy(t)
        p.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + W}" y2="{y:.1f}" class="grid"/>')
        p.append(f'<text x="{ml - 6}" y="{y + 4:.1f}" text-anchor="end" class="ax">{t:g}%</text>')
    for hl in hlines:
        if lo <= hl <= hi:
            y = sy(hl)
            p.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + W}" y2="{y:.1f}" stroke="var(--c-fault)" stroke-dasharray="4 3"/>')
            p.append(f'<text x="{ml + W}" y="{y - 3:.1f}" text-anchor="end" class="ax" fill="var(--c-fault)">{hl:g}%</text>')
    y0 = sy(0)
    p.append(f'<line x1="{ml}" y1="{y0:.1f}" x2="{ml + W}" y2="{y0:.1f}" class="axis"/>')
    for i, (v, w) in enumerate(zip(values, weather)):
        if v is None:
            continue
        y = sy(v)
        top, h = min(y, y0), abs(y - y0)
        p.append(f'<rect x="{sx(i) - bw / 2:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{max(h, 1):.1f}" fill="{wcolor.get(w, "var(--c-dim)")}" rx="1">'
                 f'<title>{escape(str(labels[i]))} {w}: {v:+.1f}%</title></rect>')
    step = max(1, n // 8)
    for i, lab in enumerate(labels):
        if i % step == 0:
            p.append(f'<text x="{sx(i):.1f}" y="{mt + H + 14}" text-anchor="middle" class="ax">{escape(str(lab)[5:])}</text>')
    x = ml
    for c, t in (("var(--c-clear)", "晴好天"), ("var(--c-cloudy)", "多云"), ("var(--c-dim)", "阴雨/缺数据")):
        p.append(f'<rect x="{x}" y="{height - 12}" width="12" height="8" fill="{c}" rx="1"/><text x="{x + 16}" y="{height - 4}" class="lg">{t}</text>')
        x += 16 + 7 * len(t) + 16
    p.append("</svg>")
    return "".join(p)


def hbar_chart(rows: List[Dict], value_key: str = "value", label_key: str = "label", unit: str = "", width: int = 420) -> str:
    """横向条形图（损失台账按站/按案例）"""
    if not rows:
        return '<p class="empty">暂无数据</p>'
    mx = max((r[value_key] or 0) for r in rows) or 1
    rh, gap, ml = 20, 7, 120
    height = len(rows) * (rh + gap) + 8
    p = [f'<svg class="chart" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" style="max-width:100%;height:auto;display:block" role="img">']
    for i, r in enumerate(rows):
        y = 4 + i * (rh + gap)
        w = max(2, (r[value_key] or 0) / mx * (width - ml - 90))
        p.append(f'<text x="{ml - 10}" y="{y + rh - 6}" text-anchor="end" class="lb">{escape(str(r[label_key]))[:22]}</text>')
        p.append(f'<rect x="{ml}" y="{y}" width="{w:.1f}" height="{rh}" rx="3" fill="{r.get("color", "var(--c-low)")}"/>')
        p.append(f'<text x="{ml + w + 8:.1f}" y="{y + rh - 6}" class="lb">{r[value_key]:,.0f}{unit}</text>')
    p.append("</svg>")
    return "".join(p)


def dev_color(dev: Optional[float], zero: bool = False, nodata: bool = False) -> str:
    """热力图配色：偏差 → 语义色"""
    if nodata:
        return "var(--c-cell-na)"
    if dev is None:
        return "var(--c-cell-sampled)"
    if zero or dev <= -0.5:
        return "var(--c-cell-dead)"
    if dev <= -0.15:
        return "var(--c-cell-bad)"
    if dev <= -0.05:
        return "var(--c-cell-low)"
    if dev <= -0.03:
        return "var(--c-cell-watch)"
    return "var(--c-cell-ok)"


def heatmap(inverters: List[Dict], max_pv: int, cell: int = 22, gap: int = 4, label_w: int = 110) -> str:
    """电站组串热力图：行 = 逆变器，列 = pv 编号；格子颜色 = 今日互比偏差，描边 = 在办案例分级。
    inverters: [{inverter_id, name, cells: {pv_index: {dev, zero, grade, string_id, cause, nodata}}}]"""
    cols = max(max_pv, 1)
    width = label_w + cols * (cell + gap) + 8
    height = 22 + len(inverters) * (cell + gap) + 6
    p = [f'<svg class="heat" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" style="max-width:100%;height:auto;display:block" role="img" aria-label="组串热力图">']
    for c in range(cols):
        if c % 2 == 0 or cols <= 16:
            p.append(f'<text x="{label_w + c * (cell + gap) + cell / 2:.1f}" y="12" text-anchor="middle" class="ax">pv{c + 1}</text>')
    for r, inv in enumerate(inverters):
        y = 20 + r * (cell + gap)
        p.append(f'<text x="{label_w - 8}" y="{y + cell - 6}" text-anchor="end" class="lb">{escape(str(inv.get("name") or inv["inverter_id"]))[:16]}</text>')
        for c in range(cols):
            x = label_w + c * (cell + gap)
            info = inv["cells"].get(c + 1)
            if info is None:
                p.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="3" class="cell-void"/>')
                continue
            fill = dev_color(info.get("dev"), info.get("zero", False), info.get("nodata", False))
            stroke = ""
            g = info.get("grade")
            if g in ("A", "B"):
                stroke = f' stroke="var(--c-{"fault" if g == "A" else "low"})" stroke-width="2.5"'
            elif g in ("C", "D", "E"):
                stroke = ' stroke="var(--c-ink)" stroke-width="1.5" stroke-dasharray="3 2"'
            dev = info.get("dev")
            tip = f'{info["string_id"]}  偏差 {dev:+.1%}' if dev is not None else f'{info["string_id"]}  {"无数据" if info.get("nodata") else "已采样，待建立基线"}'
            if g:
                tip += f'  [{g}] {info.get("cause") or ""}'
            href = f'/string/{escape(info["string_id"])}'
            p.append(f'<a href="{href}"><rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="3" fill="{fill}"{stroke}><title>{escape(tip)}</title></rect></a>')
    p.append("</svg>")
    return "".join(p)
