"""控制台模板（Jinja2，内嵌字符串，无静态目录、无外网依赖）"""

BASE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{% block title %}pvdiag{% endblock %} · 组串诊断控制台</title>
<style>
:root{
  --bg:#eef1f5;--panel:#ffffff;--panel2:#f5f7fa;--line:#d9dfe8;--ink:#17202b;--muted:#5f6b7a;--dim:#a9b3c0;
  --acc:#b85c1f;--acc-ink:#ffffff;--focus:#2d6cdf;
  --c-fault:#c0392b;--c-low:#d7860f;--c-orient:#6a4fcf;--c-ok:#1e8a5a;--c-clear:#1f8a8a;--c-cloudy:#e0a526;--c-sky:#2d6cdf;--c-ink:#17202b;--c-muted:#5f6b7a;--c-dim:#b6bfca;
  --c-cell-ok:#d2e7db;--c-cell-watch:#ffe3a3;--c-cell-low:#f7b267;--c-cell-bad:#e8734a;--c-cell-dead:#9b2226;--c-cell-na:#e3e8ee;
  --shadow:0 1px 2px rgba(23,32,43,.06);
}
@media (prefers-color-scheme:dark){:root{
  --bg:#0f141a;--panel:#161c25;--panel2:#1c242e;--line:#2a3441;--ink:#e6ebf2;--muted:#9aa5b5;--dim:#5a6678;
  --acc:#e08a4a;--acc-ink:#1a1208;--focus:#6ea0ff;
  --c-fault:#ef6b5c;--c-low:#f0a640;--c-orient:#a68cff;--c-ok:#3fb27f;--c-clear:#3fb0b0;--c-cloudy:#e9bd4e;--c-sky:#6ea0ff;--c-ink:#e6ebf2;--c-muted:#9aa5b5;--c-dim:#4a5566;
  --c-cell-ok:#234b3a;--c-cell-watch:#6b5416;--c-cell-low:#9a5a1a;--c-cell-bad:#a8402a;--c-cell-dead:#8a1e24;--c-cell-na:#262f3b;
  --shadow:none;
}}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 -apple-system,"PingFang SC","Hiragino Sans GB","Noto Sans SC","Microsoft YaHei",system-ui,sans-serif}
a{color:inherit;text-decoration:none}a:hover{text-decoration:underline}
.layout{display:grid;grid-template-columns:216px 1fr;min-height:100vh}
nav.side{background:var(--panel);border-right:1px solid var(--line);padding:18px 14px;position:sticky;top:0;height:100vh;display:flex;flex-direction:column;gap:6px}
.brand{display:flex;align-items:center;gap:10px;padding:4px 8px 14px;font-weight:700;font-size:15px;letter-spacing:.2px}
.brand .logo{width:28px;height:28px;border-radius:7px;background:linear-gradient(135deg,var(--c-low),var(--acc));display:grid;place-items:center;color:#fff;font-size:15px}
nav.side a.item{display:flex;justify-content:space-between;align-items:center;padding:8px 10px;border-radius:8px;color:var(--muted);font-weight:500}
nav.side a.item:hover{background:var(--panel2);text-decoration:none;color:var(--ink)}
nav.side a.item.on{background:var(--panel2);color:var(--ink);box-shadow:inset 3px 0 0 var(--acc)}
nav.side .cnt{font-size:12px;background:var(--line);border-radius:999px;padding:0 7px;font-variant-numeric:tabular-nums}
nav.side .cnt.hot{background:var(--c-fault);color:#fff}
nav.side .foot{margin-top:auto;color:var(--dim);font-size:12px;padding:8px 10px;line-height:1.5}
main{padding:22px 28px 48px;min-width:0}
.top{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin-bottom:18px}
.top h1{font-size:20px;margin:0;font-weight:700}
.top .sub{color:var(--muted)}
.top .right{margin-left:auto;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px 18px;box-shadow:var(--shadow);min-width:0}
.panel h2{font-size:14px;margin:0 0 12px;font-weight:600;color:var(--ink);display:flex;align-items:baseline;gap:10px}
.panel h2 .hint{color:var(--muted);font-weight:400;font-size:12px}
.grid{display:grid;gap:16px}
.cols-2{grid-template-columns:minmax(0,1fr) minmax(0,1fr);align-items:start}.cols-3-1{grid-template-columns:minmax(0,2fr) minmax(280px,1fr);align-items:start}
@media (max-width:1100px){.cols-2,.cols-3-1{grid-template-columns:1fr}}
@media (max-width:860px){.layout{grid-template-columns:1fr}nav.side{position:static;height:auto;flex-direction:row;flex-wrap:wrap;align-items:center;border-right:0;border-bottom:1px solid var(--line)}nav.side .foot,nav.side .brand{display:none}main{padding:16px}}
/* KPI 条 */
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:16px}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.kpi .v{font-size:24px;font-weight:700;font-variant-numeric:tabular-nums;line-height:1.1}
.kpi .v small{font-size:13px;font-weight:500;color:var(--muted);margin-left:2px}
.kpi .l{color:var(--muted);font-size:12px;margin-top:4px}
.kpi.fault .v{color:var(--c-fault)}.kpi.low .v{color:var(--c-low)}.kpi.ok .v{color:var(--c-ok)}
/* 表格 */
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th{text-align:left;font-weight:600;color:var(--muted);font-size:12px;padding:8px 10px;border-bottom:1px solid var(--line);background:var(--panel);position:sticky;top:0}
td{padding:9px 10px;border-bottom:1px solid var(--line);vertical-align:top}
tr:last-child td{border-bottom:0}tbody tr:hover td{background:var(--panel2)}
td.id,.id{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12.5px}
.neg{color:var(--c-fault);font-weight:600}.num{font-variant-numeric:tabular-nums;text-align:right}
.muted{color:var(--muted)}.small{font-size:12px}.dim{color:var(--dim)}
.empty{color:var(--muted);padding:18px 0;text-align:center}
/* 分级 / 状态 */
.grade{display:inline-grid;place-items:center;min-width:24px;height:24px;padding:0 7px;border-radius:6px;font-weight:700;font-size:12.5px;color:#fff;letter-spacing:.3px;white-space:nowrap}
.gA{background:var(--c-fault)}.gB{background:var(--c-low)}.gC{background:var(--c-orient)}.gD{background:var(--c-sky)}.gE{background:var(--c-muted)}.g已备案{background:var(--c-dim);color:var(--ink)}
.chip{display:inline-block;padding:1px 9px;border-radius:999px;font-size:12px;border:1px solid var(--line);color:var(--muted);background:var(--panel2);white-space:nowrap}
.chip.ok{color:var(--c-ok);border-color:color-mix(in srgb,var(--c-ok) 40%,transparent)}
.chip.warn{color:var(--c-low);border-color:color-mix(in srgb,var(--c-low) 40%,transparent)}
.chip.bad{color:var(--c-fault);border-color:color-mix(in srgb,var(--c-fault) 40%,transparent)}
.chip.info{color:var(--c-sky);border-color:color-mix(in srgb,var(--c-sky) 40%,transparent)}
tr.sev-A td:first-child{box-shadow:inset 3px 0 0 var(--c-fault)}tr.sev-B td:first-child{box-shadow:inset 3px 0 0 var(--c-low)}
tr.sev-C td:first-child{box-shadow:inset 3px 0 0 var(--c-orient)}tr.sev-D td:first-child,tr.sev-E td:first-child{box-shadow:inset 3px 0 0 var(--c-dim)}
/* 控件 */
.btn,button{appearance:none;border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:7px;padding:6px 12px;font:inherit;font-size:13px;cursor:pointer;line-height:1.3}
.btn:hover,button:hover{background:var(--panel2);text-decoration:none}
.btn.primary,button.primary{background:var(--acc);border-color:var(--acc);color:var(--acc-ink);font-weight:600}
.btn.ok,button.ok{background:var(--c-ok);border-color:var(--c-ok);color:#fff;font-weight:600}
.btn.danger,button.danger{color:var(--c-fault);border-color:color-mix(in srgb,var(--c-fault) 50%,transparent);background:var(--panel)}
.btn.sm,button.sm{padding:4px 9px;font-size:12px}
select,input[type=text],input[type=date],textarea{border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:7px;padding:6px 9px;font:inherit;font-size:13px}
textarea{width:100%;resize:vertical}
:focus-visible{outline:2px solid var(--focus);outline-offset:2px}
form.inline{display:inline-flex;gap:6px;align-items:center;flex-wrap:wrap}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden;background:var(--panel)}
.seg a{padding:5px 11px;color:var(--muted);font-size:13px;border-right:1px solid var(--line)}.seg a:last-child{border-right:0}
.seg a.on{background:var(--panel2);color:var(--ink);font-weight:600}.seg a:hover{text-decoration:none;color:var(--ink)}
/* 图 */
svg.chart .t{font-size:12px;font-weight:600;fill:var(--ink)}svg.chart .ax{font-size:10.5px;fill:var(--muted)}svg.chart .lb{font-size:11.5px;fill:var(--ink)}
svg.chart .lg{font-size:11px;fill:var(--muted)}svg.chart .grid{stroke:var(--line)}svg.chart .axis{stroke:var(--dim)}
svg.heat .ax{font-size:10px;fill:var(--muted)}svg.heat .lb{font-size:11.5px;fill:var(--ink)}svg.heat .cell-void{fill:none;stroke:var(--line);stroke-dasharray:2 2}
svg.heat a rect:hover{stroke:var(--focus);stroke-width:2.5}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--muted);align-items:center}
.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-2px;margin-right:5px}
.legend i.ring{background:none;border:2px solid var(--c-fault)}.legend i.ringb{background:none;border:2px solid var(--c-low)}
/* 解释块 */
.why{background:color-mix(in srgb,var(--c-ok) 9%,var(--panel));border-left:3px solid var(--c-ok);padding:10px 14px;border-radius:0 8px 8px 0;line-height:1.65}
dl.kv{display:grid;grid-template-columns:max-content 1fr;gap:6px 14px;margin:0}dl.kv dt{color:var(--muted)}dl.kv dd{margin:0;min-width:0}
.sticky-actions{background:var(--panel2);border-top:1px solid var(--line);padding:12px 18px;margin:16px -18px -16px;border-radius:0 0 10px 10px;display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.wo-body h3{font-size:14px;margin:14px 0 6px}.wo-body ul,.wo-body ol{margin:4px 0 8px 20px}.wo-body p{margin:4px 0}
.station-head{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;margin:0 0 8px}.station-head h3{margin:0;font-size:15px}
.alert{padding:8px 10px;border-radius:7px;background:var(--panel2);margin-bottom:6px;font-size:13px}.alert b{color:var(--c-low)}
@media (prefers-reduced-motion:no-preference){.btn,button{transition:background .12s}}
</style></head><body>
<div class="layout">
<nav class="side">
  <div class="brand"><span class="logo">☀</span><span>pvdiag<br><span class="small muted" style="font-weight:500">组串诊断控制台</span></span></div>
  <a class="item {% if active=='index' %}on{% endif %}" href="/">看板</a>
  <a class="item {% if active=='strings' %}on{% endif %}" href="/strings">组串热力图</a>
  <a class="item {% if active=='review' %}on{% endif %}" href="/review">复核队列 <span class="cnt {% if pending_n %}hot{% endif %}">{{ pending_n }}</span></a>
  <a class="item {% if active=='wo' %}on{% endif %}" href="/workorders">工单 <span class="cnt">{{ draft_n }}</span></a>
  <a class="item {% if active=='report' %}on{% endif %}" href="/reports">每日报告</a>
  <a class="item {% if active=='autonomy' %}on{% endif %}" href="/autonomy">自主运行</a>
  <div class="foot">最近巡检日 {{ latest or '—' }}<br>{{ now }}</div>
</nav>
<main>{% block body %}{% endblock %}</main>
</div>
</body></html>"""

INDEX = r"""{% extends "base.html" %}{% set active='index' %}{% block title %}看板{% endblock %}{% block body %}
<div class="top"><h1>看板</h1><span class="sub">巡检日 {{ date or '—' }}</span>
 <div class="right"><div class="seg"><a href="/" class="{% if not station %}on{% endif %}">全部电站</a>{% for s in stations %}<a href="/?station={{s.station_id}}" class="{% if station==s.station_id %}on{% endif %}">{{ s.name or s.station_id }}</a>{% endfor %}</div></div></div>

<div class="kpis">
 <div class="kpi {% if by_grade.get('A') %}fault{% endif %}"><div class="v">{{ by_grade.get('A',0) }}<small>A 故障</small></div><div class="l">待处理，直接派单</div></div>
 <div class="kpi {% if by_grade.get('B') %}low{% endif %}"><div class="v">{{ by_grade.get('B',0) }}<small>B 持续低发</small></div><div class="l">≤-5% 连续 ≥2 晴好天</div></div>
 <div class="kpi"><div class="v">{{ (by_grade.get('C',0)+by_grade.get('D',0)+by_grade.get('E',0)) }}<small>C/D/E</small></div><div class="l">档案核对 / 观察</div></div>
 <div class="kpi low"><div class="v">{{ loss.totals.lost_yuan|num(0) }}<small>元</small></div><div class="l">累计未修复损失 ≈ {{ loss.totals.lost_kwh|num(0) }} kWh</div></div>
 <div class="kpi {% if stats.false_alarm_rate is not none and stats.false_alarm_rate <= 0.1 %}ok{% endif %}"><div class="v">{% if stats.false_alarm_rate is not none %}{{ (stats.false_alarm_rate*100)|num(0) }}<small>%</small>{% else %}—{% endif %}</div><div class="l">近 30 天误报率（复核 {{ stats.n_reviewed }} 条）</div></div>
 <div class="kpi"><div class="v">{{ board.get('kpi',{}).get('detection_latency_days') if board.get('kpi',{}).get('detection_latency_days') is not none else '—' }}<small>天</small></div><div class="l">平均发现时延</div></div>
</div>

<div class="grid cols-3-1" style="align-items:start">
<div class="grid" style="gap:16px;align-content:start">
 {% for h in heat %}
 <section class="panel">
  <div class="station-head"><h3>{{ h.name }}</h3>
   <span class="chip {% if h.weather.get('weather_tag')=='clear' %}ok{% elif h.weather.get('weather_tag')=='cloudy' %}warn{% endif %}">{{ h.weather.get('weather_tag')|wcn }}{% if h.weather.get('noon_power') %} · 中午 {{ h.weather.noon_power|num(0) }} kW / 峰值参考 {{ h.weather.peak_ref|num(0) }} kW{% endif %}</span>
   <span class="muted small">{{ h.n_strings }} 路待诊断{% if h.n_unused %} · 空端口 {{ h.n_unused }}{% endif %} · 在办案例 {{ h.n_cases }}{% if h.n_dead %} · 0 电流 {{ h.n_dead }}{% endif %}</span>
   <a class="small" style="margin-left:auto" href="/strings?station={{h.station_id}}">放大 →</a></div>
  {{ h.svg|safe }}
 </section>
 {% else %}<section class="panel"><p class="empty">还没有巡检数据。导入档案并运行 <code>pvdiag run</code> 后这里会显示每个电站的组串热力图。</p></section>{% endfor %}
 <div class="legend"><span><i style="background:var(--c-cell-ok)"></i>正常 &gt;-3%</span><span><i style="background:var(--c-cell-watch)"></i>-3%~-5%</span><span><i style="background:var(--c-cell-low)"></i>-5%~-15%</span><span><i style="background:var(--c-cell-bad)"></i>&lt;-15%</span><span><i style="background:var(--c-cell-dead)"></i>0 电流</span><span><i style="background:var(--c-cell-na)"></i>无数据</span><span><i class="ring"></i>A 级在办</span><span><i class="ringb"></i>B 级在办</span><span>点格子看单串详情</span></div>

 <section class="panel"><h2>损失台账 <span class="hint">未修复案例累计电费，驱动派单优先级</span></h2>
  <div class="grid cols-2"><div><div class="small muted" style="margin-bottom:6px">按电站</div>{{ loss_svg|safe }}</div><div><div class="small muted" style="margin-bottom:6px">按案例（前 8）</div>{{ case_svg|safe }}</div></div>
 </section>
</div>

<div class="grid" style="gap:16px;align-content:start">
 <section class="panel"><h2>待复核 <span class="hint">按严重度排序</span></h2>
  {% if todo %}<table><tbody>{% for r in todo %}<tr class="sev-{{r.grade}}"><td><span class="grade g{{r.grade}}">{{r.grade}}</span></td>
  <td><a class="id" href="/suspect/{{r.suspect_id}}">{{ r.string_id or r.inverter_id ~ '（整机）' }}</a><div class="small muted">{{ r.ai_cause or r.rule_cause }}</div></td>
  <td class="num {% if r.dev_today is not none and r.dev_today < -0.05 %}neg{% endif %}">{{ r.dev_today|pct }}</td></tr>{% endfor %}</tbody></table>
  <p style="margin:10px 0 0"><a class="btn sm" href="/review">进入复核队列</a></p>
  {% else %}<p class="empty">队列为空</p>{% endif %}
 </section>
 <section class="panel"><h2>工单 <span class="hint">草稿需人工确认发送</span></h2>
  {% if wos %}<table><tbody>{% for w in wos %}<tr><td><a class="id" href="/workorder/{{w.wo_id}}">{{w.wo_id}}</a><div class="small muted">{{w.station_id}} · {{w.roof_area}} · {{w.cause}}</div></td>
  <td class="num"><span class="chip {% if w.status=='draft' %}warn{% elif w.status=='sent' %}info{% endif %}">{{ w.status|cn }}</span></td></tr>{% endfor %}</tbody></table>
  {% else %}<p class="empty">无待办工单</p>{% endif %}
 </section>
 <section class="panel"><h2>系统状态</h2>
  {% if report.agents %}<table><tbody>{% for k,v in report.agents.items() %}<tr><td class="id">{{k}}</td><td><span class="chip {% if v.status=='ok' %}ok{% elif v.status=='fail' %}bad{% else %}warn{% endif %}">{{v.status}}</span></td><td class="small muted">{{v.summary}}</td></tr>{% endfor %}</tbody></table>{% endif %}
  {% if board.get('model') %}<p class="small muted" style="margin:10px 0 0">判定模型 {{ board.model.active }} · 标注 {{ board.model.n_labels }}/{{ board.model.min_labels_for_ml }} 条 · 自动派单门控 {{ '已通过' if board.get('auto_dispatch_gate',{}).get('passed') else '未通过（' ~ board.get('auto_dispatch_gate',{}).get('reason','') ~ '）' }}</p>{% endif %}
  {% for a in alerts %}<div class="alert"><b>{{a.source}}</b> {{a.message}}</div>{% endfor %}
 </section>
</div>
</div>
{% endblock %}"""

STRINGS = r"""{% extends "base.html" %}{% set active='strings' %}{% block title %}组串热力图{% endblock %}{% block body %}
<div class="top"><h1>组串热力图</h1><span class="sub">每格一串，颜色 = 当日中午窗口互比偏差，描边 = 在办案例</span>
 <div class="right">
  <div class="seg"><a href="/strings?d={{date}}" class="{% if not station %}on{% endif %}">全部</a>{% for s in stations %}<a href="/strings?station={{s.station_id}}&d={{date}}" class="{% if station==s.station_id %}on{% endif %}">{{ s.name or s.station_id }}</a>{% endfor %}</div>
  <form class="inline" method="get"><input type="hidden" name="station" value="{{station}}"><select name="d" onchange="this.form.submit()">{% for x in dates %}<option {% if x==date %}selected{% endif %}>{{x}}</option>{% endfor %}</select></form>
 </div></div>
{% for h in heat %}<section class="panel" style="margin-bottom:16px">
 <div class="station-head"><h3>{{ h.name }}</h3><span class="chip {% if h.weather.get('weather_tag')=='clear' %}ok{% elif h.weather.get('weather_tag')=='cloudy' %}warn{% endif %}">{{ h.weather.get('weather_tag')|wcn }}</span><span class="muted small">{{ h.n_strings }} 路待诊断{% if h.n_unused %} · 空端口 {{ h.n_unused }}{% endif %} · {{ h.inverters|length }} 台逆变器 · 在办案例 {{ h.n_cases }}</span></div>
 {{ h.svg|safe }}
</section>{% else %}<section class="panel"><p class="empty">无数据</p></section>{% endfor %}
<div class="legend"><span><i style="background:var(--c-cell-ok)"></i>正常</span><span><i style="background:var(--c-cell-watch)"></i>-3%~-5%</span><span><i style="background:var(--c-cell-low)"></i>-5%~-15%</span><span><i style="background:var(--c-cell-bad)"></i>&lt;-15%</span><span><i style="background:var(--c-cell-dead)"></i>0 电流</span><span><i style="background:var(--c-cell-na)"></i>无数据</span><span><i class="ring"></i>A 级</span><span><i class="ringb"></i>B 级</span><span>虚线框 = C/D/E 观察</span></div>
{% endblock %}"""

STRING = r"""{% extends "base.html" %}{% set active='strings' %}{% block title %}{{ prof.string_id }}{% endblock %}{% block body %}
<div class="top"><h1 class="id">{{ prof.string_id }}</h1><span class="sub">{{ prof.station_id }} · {{ prof.inverter_id }} · pv{{ prof.pv_index }} · {{ prof.roof_area or '位置待补' }}</span>
 <div class="right">{% if bl %}<span class="chip {% if bl.dev_today is not none and bl.dev_today <= -0.05 %}bad{% elif bl.dev_today is not none and bl.dev_today <= -0.03 %}warn{% else %}ok{% endif %}">{{ date }} 偏差 {{ bl.dev_today|pct }} · {{ bl.weather_tag|wcn }}</span>{% endif %}
 <a class="btn sm" href="/strings?station={{prof.station_id}}">← 热力图</a></div></div>
<div class="grid cols-2">
 <section class="panel"><h2>当日电流曲线 <span class="hint">实线 = 该串，虚线 = 同群组中位</span></h2>{{ chart_svg|safe }}</section>
 <section class="panel"><h2>互比偏差趋势 <span class="hint">近 45 天，仅晴好天参与判定</span></h2>{{ trend_svg|safe }}</section>
</div>
<div class="grid cols-2" style="margin-top:16px">
 <section class="panel"><h2>档案</h2>
  <dl class="kv"><dt>串长 / 组件</dt><dd>{{ prof.n_modules }} 块 × {{ prof.module_model or '—' }} {{ prof.module_wp|num(0) }} W（Vmp {{ prof.module_vmp|num(1) }} V · Imp {{ prof.module_imp|num(2) }} A）</dd>
  <dt>朝向 / 倾角</dt><dd>{{ prof.azimuth|num(0) }}° / {{ prof.tilt|num(0) }}°</dd><dt>MPPT</dt><dd>{{ prof.mppt }}</dd>
  <dt>已知遮挡</dt><dd>{% if prof.known_shading %}是 — {{ prof.shading_note }}（预期偏差 {{ prof.shading_expected_dev|pct }}）{% else %}否{% endif %}</dd>
  <dt>互比群组</dt><dd class="id small">{{ prof.group_key }}</dd>
  {% if bl %}<dt>基线</dt><dd>自身基线比 {{ bl.baseline_ratio|num(3) }}（{{ bl.baseline_n }} 个晴好天{% if bl.cold_start %}，冷启动{% endif %}）· 群组 {{ bl.group_n }} 串 · 相对自身 {{ bl.dev_vs_own|pct }}</dd>
  <dt>持续度</dt><dd>≤-5% 连续 {{ bl.persist_days_5 }} 天 · ≤-3% 连续 {{ bl.persist_days_3 }} 天{% if bl.first_breach_date %} · 首次跌破 {{ bl.first_breach_date }}{% endif %}</dd>{% endif %}
  <dt>备注</dt><dd>{{ prof.notes or '—' }}</dd></dl>
  {% if events %}<h2 style="margin-top:14px">事件 <span class="hint">基线自事件日分段重置</span></h2><table><tbody>{% for e in events %}<tr><td>{{e.event_date}}</td><td>{{e.event_type}}</td><td class="small muted">{{e.notes}}</td></tr>{% endfor %}</tbody></table>{% endif %}
 </section>
 <section class="panel"><h2>案例</h2>
  {% if cases %}<table><thead><tr><th>分级</th><th>案例</th><th>原因</th><th>状态</th></tr></thead><tbody>{% for c in cases %}<tr class="sev-{{c.grade}}"><td><span class="grade g{{c.grade}}">{{c.grade}}</span></td><td><a class="id" href="/suspect/{{c.suspect_id}}">{{c.suspect_id}}</a><div class="small muted">{{c.first_date}} → {{c.date}} · 持续 {{c.persist_days}} 天</div></td><td>{{ c.ai_cause or c.rule_cause }}</td><td><span class="chip">{{ c.status|cn }}</span></td></tr>{% endfor %}</tbody></table>
  {% else %}<p class="empty">该串没有过嫌疑案例</p>{% endif %}
  {% if sigs %}<h2 style="margin-top:14px">近 10 日签名</h2><table><thead><tr><th>日期</th><th>天气</th><th class="num">比值均值</th><th class="num">标准差</th><th>凹陷</th><th class="num">电压偏差</th></tr></thead><tbody>
  {% for g in sigs %}<tr><td>{{g.date}}</td><td>{{g.weather_tag|wcn}}</td><td class="num">{{g.ratio_mean|num(3)}}</td><td class="num">{{g.ratio_std|num(3)}}</td><td>{% if g.dip_depth %}{{g.dip_start}}–{{g.dip_end}} 深 {{(g.dip_depth*100)|num(0)}}%{% else %}—{% endif %}</td><td class="num">{{g.voltage_offset|pct}}</td></tr>{% endfor %}</tbody></table>{% endif %}
 </section>
</div>
{% endblock %}"""

REVIEW = r"""{% extends "base.html" %}{% set active='review' %}{% block title %}复核队列{% endblock %}{% block body %}
<div class="top"><h1>复核队列</h1><span class="sub">{{ rows|length }} 条 · 一屏看全，清单式批量复核</span>
 <div class="right">
  <div class="seg"><a href="/review?station={{station}}" class="{% if not grade %}on{% endif %}">全部 {{ counts.values()|sum }}</a>{% for g in ['A','B','C','D','E'] %}<a href="/review?station={{station}}&grade={{g}}" class="{% if g==grade %}on{% endif %}">{{g}} {{ counts.get(g,0) }}</a>{% endfor %}</div>
  <div class="seg"><a href="/review?grade={{grade}}" class="{% if not station %}on{% endif %}">全部电站</a>{% for s in stations %}<a href="/review?station={{s.station_id}}&grade={{grade}}" class="{% if s.station_id==station %}on{% endif %}">{{ s.name or s.station_id }}</a>{% endfor %}</div>
 </div></div>
<section class="panel" style="padding:0;overflow:auto">
<table><thead><tr><th>分级</th><th>对象</th><th class="num">偏差</th><th class="num">持续</th><th>规则 → AI 判定</th><th>首报</th><th>动作</th><th style="min-width:330px">复核</th></tr></thead><tbody>
{% for r in rows %}<tr class="sev-{{r.grade}}">
<td><span class="grade g{{r.grade}}">{{r.grade}}</span></td>
<td><a class="id" href="/suspect/{{r.suspect_id}}">{{ r.string_id if r.string_id else r.inverter_id ~ '（整机）' }}</a><div class="small muted">{{r.station_id}}{% if r.low_confidence %} · <span class="chip warn">低置信</span>{% endif %}</div></td>
<td class="num {% if r.dev_today is not none and r.dev_today <= -0.05 %}neg{% endif %}">{{ r.dev_today|pct }}</td><td class="num">{{r.persist_days}} 天</td>
<td>{{r.rule_cause}}<div class="small muted">AI：<b style="color:var(--ink)">{{r.ai_cause or '—'}}</b> {{ (r.confidence*100)|num(0) ~ '%' if r.confidence is not none else '' }}</div></td>
<td class="small">{{r.first_date}}</td><td class="small">{{r.action or '—'}}</td>
<td><form class="inline" method="post" action="/review/{{r.suspect_id}}/confirm"><input type="hidden" name="back" value="/review?station={{station}}&grade={{grade}}">
<select name="true_cause" aria-label="真实原因"><option value="">原因同判定</option>{% for c in causes %}<option>{{c}}</option>{% endfor %}</select><button class="ok sm">确认异常</button></form>
<form class="inline" method="post" action="/review/{{r.suspect_id}}/reject" style="margin-top:4px"><input type="hidden" name="back" value="/review?station={{station}}&grade={{grade}}">
<select name="reason" aria-label="驳回原因">{% for x in reasons %}<option>{{x}}</option>{% endfor %}</select><button class="danger sm">驳回</button></form></td>
</tr>{% else %}<tr><td colspan="8" class="empty">队列为空。下一次巡检后新的嫌疑会出现在这里。</td></tr>{% endfor %}</tbody></table>
</section>
<p class="small muted" style="margin-top:10px">确认 = 真实异常（可改原因）；驳回 = 误报（选原因）。两者都写入标注库，是 AI 模型的训练数据。近 30 天已复核 {{ stats.n_reviewed }} 条{% if stats.confirm_rate is not none %}，确认率 {{ (stats.confirm_rate*100)|num(0) }}%{% endif %}。</p>
{% endblock %}"""

SUSPECT = r"""{% extends "base.html" %}{% set active='review' %}{% block title %}{{ s.string_id or s.inverter_id }}{% endblock %}{% block body %}
<div class="top"><span class="grade g{{s.grade}}" style="height:28px;min-width:28px;font-size:14px">{{s.grade}}</span><h1 class="id">{{ s.string_id or s.inverter_id ~ '（整机）' }}</h1>
 <span class="sub">{{ s.station_id }} · 案例 {{ s.suspect_id }}</span>
 <div class="right"><span class="chip {% if s.status in ['confirmed','dispatched'] %}info{% elif s.status=='rejected' %}bad{% elif s.status=='closed' %}ok{% else %}warn{% endif %}">{{ s.status|cn }}</span>
 {% if s.string_id %}<a class="btn sm" href="/string/{{s.string_id}}">单串详情</a>{% endif %}<a class="btn sm" href="/review">← 队列</a>{% if next_id %}<a class="btn sm" href="/suspect/{{next_id}}">下一条 →</a>{% endif %}</div></div>

<div class="grid cols-2">
 <section class="panel"><h2>当日电流曲线 <span class="hint">{% if s.scope=='inverter' %}该逆变器各串{% else %}实线 = 该串，虚线 = 同群组中位{% endif %}</span></h2>{{ chart_svg|safe }}</section>
 <section class="panel"><h2>互比偏差趋势 <span class="hint">近 45 天，仅晴好天参与判定</span></h2>{{ trend_svg|safe }}</section>
</div>
<div class="grid cols-2" style="margin-top:16px">
<section class="panel"><h2>判定</h2>
 <p class="why" style="margin:0 0 12px">{{ s.explanation or '尚无 AI 判定（等待下一次巡检）' }}</p>
 <details {% if analysis and analysis.source != 'template' %}open{% endif %} style="margin:0 0 12px"><summary class="small muted" style="cursor:pointer">分析助手 <span class="chip {% if analysis.source.startswith('llm') %}ok{% else %}{% endif %}">{{ '大模型 ' ~ analysis.source[4:] if analysis.source.startswith('llm') else '模板（未配置大模型）' }}</span></summary>
  <div class="small" style="white-space:pre-wrap;line-height:1.7;margin-top:8px">{{ analysis.text }}</div>
  <form class="inline" method="post" action="/suspect/{{s.suspect_id}}/ask" style="margin-top:8px"><input type="text" name="question" placeholder="追问：损失多少 / 为什么这样判 / 同类历史 / 现场怎么办" style="min-width:320px" required><button class="sm">问</button></form>
  {% for q in qa %}<div class="alert small" style="white-space:pre-wrap">{{ q.note }}</div>{% endfor %}
 </details>
 <dl class="kv">
  <dt>规则原因</dt><dd>{{ s.rule_cause }}</dd>
  <dt>AI 原因</dt><dd><b>{{ s.ai_cause or '—' }}</b>{% if s.confidence is not none %} · 置信度 {{ (s.confidence*100)|num(0) }}%{% endif %} <span class="small muted">{{ s.model_version or '' }}</span></dd>
  <dt>建议动作</dt><dd>{{ s.action or '—' }}</dd>
  <dt>偏差 / 持续</dt><dd>{{ s.dev_today|pct }} · {{ s.persist_days }} 天 · 首报 {{ s.first_date }} · 最近 {{ s.date }}</dd>
  {% if proba %}<dt>概率</dt><dd class="small">{% for k,v in proba.items() %}{{k}} {{ (v*100)|num(0) }}%　{% endfor %}</dd>{% endif %}
  <dt>签名</dt><dd class="small">{% for n in sig.get('notes',[]) %}{{n}}<br>{% endfor %}{% if sig.dip_depth %}凹陷 {{ sig.dip_start }}–{{ sig.dip_end }}，深度 {{ (sig.dip_depth*100)|num(0) }}%，其余时段比值 {{ sig.outside_dip_ratio|num(2) }}<br>{% endif %}{% if sig.overcast_dev is not none %}阴天偏差 {{ sig.overcast_dev|pct }}{% endif %}</dd>
  <dt>损失</dt><dd>累计 {{ loss.lost_kwh|num(0) }} kWh ≈ <b>{{ loss.lost_yuan|num(0) }} 元</b>（{{ loss.days }} 个晴好天，约 {{ loss.per_day_kwh|num(1) }} kWh/天）</dd>
  {% if wo %}<dt>工单</dt><dd>{% for w in wo %}<a class="id" href="/workorder/{{w.wo_id}}">{{w.wo_id}}</a> <span class="chip">{{ w.status|cn }}</span> {% endfor %}</dd>{% endif %}
 </dl>
 {% if s.status in ['open','reviewing','confirmed','dispatched'] %}
 <div class="sticky-actions">
  <form class="inline" method="post" action="/review/{{s.suspect_id}}/confirm"><input type="hidden" name="back" value="{% if next_id %}/suspect/{{next_id}}{% else %}/review{% endif %}">
   <select name="true_cause"><option value="">原因同判定（{{ s.ai_cause or s.rule_cause }}）</option>{% for c in causes %}<option>{{c}}</option>{% endfor %}</select>
   <input type="text" name="note" placeholder="备注（可选）" size="18"><button class="ok">确认异常</button></form>
  <form class="inline" method="post" action="/review/{{s.suspect_id}}/reject"><input type="hidden" name="back" value="{% if next_id %}/suspect/{{next_id}}{% else %}/review{% endif %}">
   <select name="reason">{% for x in reasons %}<option>{{x}}</option>{% endfor %}</select><button class="danger">驳回（误报）</button></form>
 </div>{% endif %}
</section>
<section class="panel"><h2>{% if prof %}组串档案{% else %}整机级嫌疑{% endif %}</h2>
 {% if prof %}<dl class="kv">
  <dt>逆变器 / MPPT / pv</dt><dd>{{prof.inverter_id}} / {{prof.mppt}} / pv{{prof.pv_index}}</dd>
  <dt>串长 / 组件</dt><dd>{{prof.n_modules}} 块 × {{prof.module_model or '—'}} {{prof.module_wp|num(0)}} W（Vmp {{prof.module_vmp|num(1)}} V · Imp {{prof.module_imp|num(2)}} A）</dd>
  <dt>朝向 / 倾角</dt><dd>{{prof.azimuth|num(0)}}° / {{prof.tilt|num(0)}}°</dd><dt>屋面位置</dt><dd>{{prof.roof_area or '待补'}}</dd>
  <dt>已知遮挡</dt><dd>{% if prof.known_shading %}是 — {{ prof.shading_note or '' }}{% else %}否{% endif %}</dd>
  <dt>互比群组</dt><dd class="id small">{{prof.group_key}}</dd><dt>备注</dt><dd>{{prof.notes or '—'}}</dd></dl>
 {% else %}<p class="muted">机间互比发现整机参照偏低：各串中位电流 / 组件 Imp 低于站内其它逆变器。优先查 MPPT 告警与限功率设置。</p>{% endif %}
 {% if reviews %}<h2 style="margin-top:14px">复核记录</h2><table><tbody>{% for r in reviews %}<tr><td class="small muted">{{r.reviewed_at[:16]}}</td><td>{{r.reviewer}}</td><td><b>{{ {'confirm':'确认','modify':'确认并改原因','reject':'驳回','auto_close':'自愈关闭'}.get(r.decision, r.decision) }}</b></td><td>{{r.true_cause or ''}} {{r.reject_reason or ''}}</td><td class="small muted">{{r.note or ''}}</td></tr>{% endfor %}</tbody></table>{% endif %}
</section>
</div>
{% endblock %}"""

WORKORDERS = r"""{% extends "base.html" %}{% set active='wo' %}{% block title %}工单{% endblock %}{% block body %}
<div class="top"><h1>工单</h1><span class="sub">草稿 → 已派发 → 待复查 → 复查归零 / 重开</span>
 <div class="right"><div class="seg"><a href="/workorders" class="{% if not status %}on{% endif %}">全部</a>{% for x in ['draft','sent','done','verified','reopened'] %}<a href="/workorders?status={{x}}" class="{% if x==status %}on{% endif %}">{{ x|cn }} {{ counts.get(x,0) }}</a>{% endfor %}</div></div></div>
<section class="panel" style="padding:0;overflow:auto">
<table><thead><tr><th>工单</th><th>电站 / 区域</th><th>原因</th><th class="num">优先级</th><th class="num">损失/天</th><th class="num">累计损失</th><th>状态</th><th>操作</th></tr></thead><tbody>
{% for w in rows %}<tr><td><a class="id" href="/workorder/{{w.wo_id}}">{{w.wo_id}}</a><div class="small muted">{{ w.created_at[:10] }}</div></td><td>{{w.station_id}}<div class="small muted">{{w.roof_area}}</div></td><td>{{w.cause}}</td>
<td class="num">{{w.priority|num(1)}}</td><td class="num">{{w.loss_kwh_per_day|num(1)}} kWh</td><td class="num">{{w.loss_yuan_total|num(0)}} 元</td>
<td><span class="chip {% if w.status=='draft' %}warn{% elif w.status=='verified' %}ok{% elif w.status=='reopened' %}bad{% elif w.status=='sent' %}info{% endif %}">{{ w.status|cn }}</span></td>
<td>{% if w.status in ['draft','reopened'] %}<form class="inline" method="post" action="/workorder/{{w.wo_id}}/send"><button class="primary sm">确认发送</button></form>{% else %}<a class="btn sm" href="/workorder/{{w.wo_id}}">查看</a>{% endif %}</td></tr>
{% else %}<tr><td colspan="8" class="empty">还没有工单。复核确认的异常和高置信 A/B 级会自动生成草稿。</td></tr>{% endfor %}</tbody></table>
</section>
<p class="small muted" style="margin-top:10px">优先级 = 日损失电量 × 电价 × 预计持续天数 ÷ 修复成本；同屋面多串合并为一张区域工单。</p>
{% endblock %}"""

WORKORDER = r"""{% extends "base.html" %}{% set active='wo' %}{% block title %}{{ wo.wo_id }}{% endblock %}{% block body %}
<div class="top"><h1 class="id">{{ wo.wo_id }}</h1><span class="sub">{{ wo.station_id }} · {{ wo.roof_area }} · 优先级 {{ wo.priority|num(1) }}</span>
 <div class="right"><span class="chip {% if wo.status=='draft' %}warn{% elif wo.status=='verified' %}ok{% elif wo.status=='reopened' %}bad{% elif wo.status=='sent' %}info{% endif %}">{{ wo.status|cn }}</span><span class="small muted">{{ wo.channel }} {{ wo.dispatch_result or '' }}</span><a class="btn sm" href="/workorders">← 工单</a></div></div>
<div class="grid cols-2">
 <section class="panel wo-body">{{ body_html|safe }}
  <p class="small muted" style="margin-top:12px">涉及案例：{% for s in suspect_ids %}<a class="id" href="/suspect/{{s}}">{{s}}</a> {% endfor %}</p></section>
 <section class="panel">
  {% if wo.status in ['draft','reopened'] %}<h2>派发</h2><p class="muted small">渠道 {{ wo.channel }}（dryrun 写入 data/outbox/；配置企业微信 webhook 后直接推送到群）。</p>
  <form method="post" action="/workorder/{{wo.wo_id}}/send"><button class="primary">确认发送工单</button></form><hr style="border:0;border-top:1px solid var(--line);margin:16px 0">{% endif %}
  {% if wo.status in ['sent','reopened','draft'] %}<h2>现场结果回填</h2>
  <form method="post" action="/workorder/{{wo.wo_id}}/close" class="grid" style="gap:10px">
   <label>实际原因<br><input type="text" name="field_cause" required placeholder="如：MC4 接头烧蚀" style="width:100%"></label>
   <div class="grid cols-2" style="gap:10px"><label>处理动作<br><select name="field_action" style="width:100%">{% for a in ['维修','清洗','更换','重接线','换逆变器','无需处理'] %}<option>{{a}}</option>{% endfor %}</select></label>
   <label>处理日期<br><input type="date" name="done_date" style="width:100%"></label></div>
   <label>备注<br><textarea name="note" rows="3" placeholder="若为巡检发现而系统未报，请写明“漏报”，用于统计漏报率"></textarea></label>
   <button class="ok">回填并登记事件</button>
   <p class="small muted" style="margin:0">回填后基线自处理日重置，{{ recheck_days }} 个晴好天后自动复查：归零 → 复查归零；未归零 → 自动重开。</p></form>
  {% elif wo.status=='done' %}<h2>已回填，等待复查</h2><dl class="kv"><dt>实际原因</dt><dd>{{ wo.field_cause }}</dd><dt>处理动作</dt><dd>{{ wo.field_action }}</dd><dt>备注</dt><dd>{{ wo.field_note or '—' }}</dd><dt>复查到期</dt><dd>{{ wo.recheck_due }}</dd></dl>
  {% else %}<h2>复查结果</h2><p>{{ wo.recheck_result }}</p><dl class="kv"><dt>实际原因</dt><dd>{{ wo.field_cause }}</dd><dt>处理动作</dt><dd>{{ wo.field_action }}</dd></dl>{% endif %}
 </section>
</div>
{% endblock %}"""

REPORT = r"""{% extends "base.html" %}{% set active='report' %}{% block title %}每日报告{% endblock %}{% block body %}
<div class="top"><h1>每日巡检报告</h1><span class="sub">{{ date or '—' }}</span>
 <div class="right"><form class="inline" method="get" action="/reports" onsubmit="location.href='/report/'+this.d.value;return false"><select name="d" onchange="location.href='/report/'+this.value">{% for x in dates %}<option {% if x==date %}selected{% endif %}>{{x}}</option>{% endfor %}</select></form></div></div>
{% if not r %}<section class="panel"><p class="empty">还没有报告。运行 <code>pvdiag run</code> 后生成。</p></section>{% else %}
<div class="grid cols-3-1">
<div class="grid" style="gap:16px">
 <section class="panel"><h2>嫌疑清单 <span class="hint">{{ r.suspects|length }} 条</span></h2>
  <table><thead><tr><th>分级</th><th>对象</th><th class="num">偏差</th><th class="num">持续</th><th>规则 → AI</th><th>动作</th><th>状态</th></tr></thead><tbody>
  {% for s in r.suspects %}<tr class="sev-{{s.grade}}"><td><span class="grade g{{s.grade}}">{{s.grade}}</span></td><td><a class="id" href="/suspect/{{s.suspect_id}}">{{ s.string_id if s.string_id else s.inverter_id ~ '（整机）' }}</a></td>
  <td class="num">{{ s.dev_today|pct }}</td><td class="num">{{s.persist_days}} 天</td><td>{{s.rule_cause}} → {{ s.ai_cause or '—' }} {{ (s.confidence*100)|num(0) ~ '%' if s.confidence is not none else '' }}</td><td class="small">{{s.action or '—'}}</td><td><span class="chip">{{ s.status|cn }}</span></td></tr>
  {% else %}<tr><td colspan="7" class="empty">当日无嫌疑</td></tr>{% endfor %}</tbody></table>
  {% if r.suspects %}<details style="margin-top:10px"><summary class="small muted" style="cursor:pointer">判定依据（为什么）</summary><ul class="small" style="margin:8px 0 0 18px;line-height:1.7">{% for s in r.suspects if s.explanation %}<li><b class="id">{{ s.string_id or s.inverter_id }}</b>：{{ s.explanation }}</li>{% endfor %}</ul></details>{% endif %}
 </section>
 {% if r.region and r.region.packages %}<section class="panel"><h2>区域任务包 <span class="hint">该去哪修什么</span></h2><table><thead><tr><th>电站 / 区域</th><th class="num">串数</th><th>原因</th><th class="num">损失/天</th><th class="num">优先级</th></tr></thead><tbody>
  {% for p in r.region.packages %}<tr><td>{{p.station_id}} · {{p.roof_area}}</td><td class="num">{{p.n_strings}}</td><td class="small">{% for k,v in p.causes.items() %}{{k}} {{v}}{% if not loop.last %} · {% endif %}{% endfor %}</td><td class="num">{{p.loss_kwh_per_day|num(1)}} kWh</td><td class="num">{{p.priority}}</td></tr>{% endfor %}</tbody></table></section>{% endif %}
 {% if r.region and r.region.dust_candidates %}<section class="panel"><h2>清洗建议 <span class="hint">积灰签名：慢渐变 + 雨后恢复</span></h2><table><tbody>{% for d in r.region.dust_candidates %}<tr><td class="id">{{d.string_id}}</td><td>{{d.pattern}}</td><td class="num">{{ d.dev_now|pct }}</td><td class="small muted">{{d.advice}}</td></tr>{% endfor %}</tbody></table></section>{% endif %}
</div>
<div class="grid" style="gap:16px;align-content:start">
 <section class="panel"><h2>天气 / 晴好天判定</h2>{% for w in r.weather %}<div style="margin-bottom:6px"><b>{{w.station_id}}</b> <span class="chip {% if w.weather_tag=='clear' %}ok{% elif w.weather_tag=='cloudy' %}warn{% endif %}">{{ w.weather_tag|wcn }}</span> <span class="small muted">中午 {{ w.noon_power|num(0) }} kW / 峰值参考 {{ w.peak_ref|num(0) }} kW</span></div>{% endfor %}</section>
 <section class="panel"><h2>智能体运行</h2><table><tbody>{% for k,v in r.agents.items() %}<tr><td class="id">{{k}}</td><td><span class="chip {% if v.status=='ok' %}ok{% elif v.status=='fail' %}bad{% else %}warn{% endif %}">{{v.status}}</span></td><td class="small muted">{{v.summary}}</td></tr>{% endfor %}</tbody></table></section>
 {% if r.loss and r.loss.totals %}<section class="panel"><h2>损失台账</h2><div class="kpi low" style="border:0;padding:0"><div class="v">{{ r.loss.totals.lost_yuan|num(0) }}<small>元</small></div><div class="l">累计未修复 ≈ {{ r.loss.totals.lost_kwh|num(0) }} kWh</div></div></section>{% endif %}
 {% if r.review and r.review.advice %}<section class="panel"><h2>阈值调优建议</h2><p class="small">{{ r.review.advice }}</p></section>{% endif %}
 {% if r.alerts %}<section class="panel"><h2>告警</h2>{% for a in r.alerts %}<div class="alert"><b>{{a.source}}</b> {{a.message}}</div>{% endfor %}</section>{% endif %}
</div></div>{% endif %}
{% endblock %}"""


AUTONOMY = r"""{% extends "base.html" %}{% set active='autonomy' %}{% block title %}自主运行{% endblock %}{% block body %}
<div class="top"><h1>自主运行</h1><span class="sub">系统自己在调什么、为什么、怎么撤回</span></div>
<div class="grid cols-3-1">
<div class="grid" style="gap:16px">
 <section class="panel"><h2>派单门控 <span class="hint">配置 {{ cfg_mode }} · 连续 {{ gate.streak or 0 }}/{{ open_days }} 天达标后放开，恶化立即收回</span></h2>
  <p><span class="chip {% if gate.effective=='auto' %}ok{% else %}warn{% endif %}">当前 {{ '自动派单' if gate.effective=='auto' else '草稿待人工确认' }}</span>
   <span class="small muted">{{ gate.reason or '' }}{% if gate.since %} · 自 {{ gate.since }}{% endif %}</span></p>
  {% if gate.history %}<table><tbody>{% for h in gate.history|reverse %}<tr><td class="small muted" style="width:90px">{{h.date}}</td><td class="small">{{h.event}}</td></tr>{% endfor %}</tbody></table>{% else %}<p class="empty">尚无开合记录</p>{% endif %}
 </section>
 <section class="panel"><h2>阈值自调参 <span class="hint">{{ autotune.mode }} · 影子回放考试达标才生效 · 生效后确认率下跌自动回滚</span></h2>
  <p class="small">当前运行口径：{% for k,v in current.items() %}<code>{{k.split('.')[-1]}}={{v}}</code> {% endfor %}
   {% if overrides %}<span class="chip info">覆盖层生效中</span>{% else %}<span class="chip">使用 config.yaml 口径</span>{% endif %}</p>
  {% if tuning_decision %}<div class="why small">{{ tuning_decision }}</div>{% endif %}
  <table style="margin-top:10px"><thead><tr><th>日期</th><th>参数</th><th>变更</th><th>状态</th><th>原因</th></tr></thead><tbody>
  {% for c in changes %}<tr><td class="small">{{c.run_date}}</td><td class="id">{{c.param.split('.')[-1]}}</td><td>{{c.old_value}} → {{c.new_value}}</td>
   <td><span class="chip {% if c.status=='applied' %}ok{% elif c.status=='rolled_back' %}bad{% else %}info{% endif %}">{{ {'applied':'已生效','rolled_back':'已回滚','proposed':'提议','deferred':'冷却中'}.get(c.status, c.status) }}</span></td><td class="small muted">{{c.reason}}</td></tr>
  {% else %}<tr><td colspan="5" class="empty">尚无参数变更（复核样本够了才会动）</td></tr>{% endfor %}</tbody></table>
 </section>
 <section class="panel"><h2>模型生命周期 <span class="hint">按时间留出最近 25% 标注考试，不低于规则代理/现役才上线；线上一致率落后 → 自动降级</span></h2>
  {% if live %}<p class="small">近 30 天与复核结论一致率：模型 <b>{{ (live.model*100)|num(0) }}%</b> vs 规则代理 <b>{{ (live.rule_proxy*100)|num(0) }}%</b>（n={{ live.n }}）</p>{% endif %}
  <table><thead><tr><th>版本</th><th>训练于</th><th class="num">标注数</th><th>状态</th><th>留出集考试（挑战者 / 规则 / 现役）</th></tr></thead><tbody>
  {% for m in models %}<tr><td class="id">{{m.model_version}}</td><td class="small">{{m.trained_at[:16]}}</td><td class="num">{{m.n_labels}}</td>
   <td><span class="chip {% if m.status=='active' %}ok{% elif m.status in ('rejected','demoted') %}bad{% else %}{% endif %}">{{ {'active':'现役','rejected':'未过考','retired':'已退役','demoted':'已降级'}.get(m.status or '', m.status or '—') }}</span></td>
   <td class="small muted">{{ m.exam }}</td></tr>
  {% else %}<tr><td colspan="5" class="empty">规则代理（标注 {{ n_labels }} / {{ min_labels }} 条，够了自动训练）</td></tr>{% endfor %}</tbody></table>
 </section>
</div>
<div class="grid" style="gap:16px;align-content:start">
 <section class="panel"><h2>健康自检 <span class="hint">{{ health_date or '—' }}</span></h2>
  {% for c in health %}<div class="alert" style="display:flex;gap:8px"><span class="chip {% if c.status=='ok' %}ok{% elif c.status=='fail' %}bad{% else %}warn{% endif %}">{{c.status}}</span><span class="small"><b style="color:inherit">{{c.name}}</b> {{c.detail}}</span></div>{% else %}<p class="empty">跑过一次巡检后出现</p>{% endfor %}
 </section>
 <section class="panel"><h2>数据接入自愈</h2>{% for i in ingest %}<div class="small" style="margin-bottom:6px"><b>{{i.station_id}}</b> 最后数据 {{ i.last or '—' }}{% if i.failures %} · <span class="chip bad">连续失败 {{i.failures}}</span>{% endif %}</div>{% else %}<p class="empty">无电站</p>{% endfor %}</section>
 <section class="panel"><h2>已推送的关键事件</h2>{% for a in pushed %}<div class="alert small"><b>{{a.run_date}}</b> {{a.message}}</div>{% else %}<p class="empty">尚无推送</p>{% endfor %}</section>
</div></div>
{% endblock %}"""

TEMPLATES = {"base.html": BASE, "autonomy.html": AUTONOMY, "index.html": INDEX, "strings.html": STRINGS, "string.html": STRING, "review.html": REVIEW,
             "suspect.html": SUSPECT, "workorders.html": WORKORDERS, "workorder.html": WORKORDER, "report.html": REPORT}
