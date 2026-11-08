"""复核界面模板（Jinja2，内嵌字符串，无需静态目录）"""

BASE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{% block title %}pvdiag{% endblock %} · 组串异常复核</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1d2433;--muted:#6b7280;--line:#e5e7eb;--a:#0f766e;--b:#b45309;--c:#6d28d9;--d:#2563eb;--e:#6b7280;--ok:#15803d;--bad:#b91c1c}
*{box-sizing:border-box}body{margin:0;font:14px/1.5 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif;background:var(--bg);color:var(--ink)}
header{background:#111827;color:#fff;padding:10px 20px;display:flex;gap:18px;align-items:center}header a{color:#d1d5db;text-decoration:none}header a:hover{color:#fff}
header .brand{font-weight:700;color:#fff}main{max-width:1280px;margin:0 auto;padding:18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin-bottom:14px}
.grid{display:grid;gap:14px}.g3{grid-template-columns:repeat(auto-fit,minmax(220px,1fr))}.g2{grid-template-columns:1fr 1fr}
.kpi .v{font-size:26px;font-weight:700}.kpi .l{color:var(--muted);font-size:12px}
table{width:100%;border-collapse:collapse}th,td{padding:7px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;font-size:13px}th{color:var(--muted);font-weight:600;background:#fafafa}
.badge{display:inline-block;padding:1px 8px;border-radius:999px;color:#fff;font-size:12px;font-weight:600}
.gA{background:var(--bad)}.gB{background:var(--b)}.gC{background:var(--c)}.gD{background:var(--d)}.gE{background:var(--e)}.g已备案{background:#9ca3af}
.muted{color:var(--muted)}.neg{color:var(--bad);font-weight:600}.pos{color:var(--ok)}
button,.btn{border:1px solid var(--line);background:#fff;border-radius:6px;padding:5px 10px;cursor:pointer;font-size:13px}
button.ok{background:var(--ok);color:#fff;border-color:var(--ok)}button.no{background:#fff;color:var(--bad);border-color:var(--bad)}button.pri{background:#111827;color:#fff;border-color:#111827}
select,input[type=text],input[type=date],textarea{border:1px solid var(--line);border-radius:6px;padding:5px 8px;font-size:13px}
form.inline{display:inline-flex;gap:6px;align-items:center;flex-wrap:wrap}
pre{white-space:pre-wrap;background:#f9fafb;border:1px solid var(--line);border-radius:8px;padding:12px;font-size:13px}
.expl{background:#f0fdf4;border-left:4px solid var(--ok);padding:8px 12px;border-radius:6px}
.small{font-size:12px}h2{font-size:17px;margin:0 0 10px}h3{font-size:15px;margin:0 0 8px}
@media (max-width:800px){.g2{grid-template-columns:1fr}}
</style></head><body>
<header><span class="brand">⚡ pvdiag</span><a href="/">看板</a><a href="/review">复核队列</a><a href="/workorders">工单</a>
<span style="margin-left:auto" class="small muted">{{ now }}</span></header>
<main>{% block body %}{% endblock %}</main>
</body></html>"""

INDEX = r"""{% extends "base.html" %}{% block title %}看板{% endblock %}{% block body %}
<h2>运营看板 <span class="muted small">最近巡检日 {{ date or '-' }}</span></h2>
<div class="grid g3">
 <div class="card kpi"><div class="v">{{ pending }}</div><div class="l">待复核（{% for g,n in by_grade.items() %}{{g}}={{n}} {% endfor %}）</div></div>
 <div class="card kpi"><div class="v">{{ wos.get('draft',0) }}</div><div class="l">工单草稿待确认 · 已发 {{ wos.get('sent',0) }} · 待复查 {{ wos.get('done',0) }} · 重开 {{ wos.get('reopened',0) }}</div></div>
 <div class="card kpi"><div class="v">{{ loss.totals.lost_kwh|num(0) }} kWh</div><div class="l">累计未修复损失 ≈ {{ loss.totals.lost_yuan|num(0) }} 元（钱在漏）</div></div>
 <div class="card kpi"><div class="v">{{ stats.false_alarm_rate if stats.false_alarm_rate is not none else '-' }}</div><div class="l">近 30 天误报率（复核 {{ stats.n_reviewed }} 条）</div></div>
 <div class="card kpi"><div class="v">{{ board.get('kpi',{}).get('detection_latency_days','-') }} 天</div><div class="l">平均发现时延（起算日→首次报警）</div></div>
 <div class="card kpi"><div class="v">{{ board.get('model',{}).get('active','rule-proxy-v1') }}</div><div class="l">当前判定模型 · 标注 {{ board.get('model',{}).get('n_labels',0) }} 条 · 自动派单门控 {{ '通过' if board.get('auto_dispatch_gate',{}).get('passed') else '未通过' }}</div></div>
</div>
<div class="grid g2">
<div class="card"><h3>各站损失台账</h3><table><tr><th>电站</th><th>损失电量</th><th>损失电费</th></tr>
{% for st,v in loss.by_station.items() %}<tr><td>{{st}}</td><td>{{v.lost_kwh|num(0)}} kWh</td><td>{{v.lost_yuan|num(0)}} 元</td></tr>{% else %}<tr><td colspan=3 class="muted">无</td></tr>{% endfor %}</table></div>
<div class="card"><h3>告警</h3><table>{% for a in alerts %}<tr><td class="small muted">{{a.run_date}}</td><td><b>{{a.source}}</b></td><td>{{a.message}}</td></tr>{% else %}<tr><td class="muted">无</td></tr>{% endfor %}</table></div>
</div>
{% if report.agents %}<div class="card"><h3>最近一次流水线 {{ report.date }}</h3><table>
{% for k,v in report.agents.items() %}<tr><td>{{ '✅' if v.status=='ok' else '⚠️' }}</td><td><code>{{k}}</code></td><td>{{v.summary}}</td></tr>{% endfor %}</table>
<p class="small muted">Markdown 报告：data/reports/{{ report.date }}/report.md</p></div>{% endif %}
{% endblock %}"""

REVIEW = r"""{% extends "base.html" %}{% block title %}复核队列{% endblock %}{% block body %}
<h2>复核队列 <span class="muted small">{{ rows|length }} 条 · 清单式批量复核，一人一天几百串</span></h2>
<form class="inline card" method="get">电站 <select name="station"><option value="">全部</option>{% for s in stations %}<option {% if s==station %}selected{% endif %}>{{s}}</option>{% endfor %}</select>
分级 <select name="grade"><option value="">全部</option>{% for g in ['A','B','C','D','E'] %}<option {% if g==grade %}selected{% endif %}>{{g}}</option>{% endfor %}</select><button>筛选</button></form>
<div class="card"><table><tr><th>分级</th><th>对象</th><th>偏差</th><th>持续</th><th>规则原因 → AI 原因（置信）</th><th>首报</th><th>动作</th><th>复核</th></tr>
{% for r in rows %}<tr>
<td><span class="badge g{{r.grade}}">{{r.grade}}</span></td>
<td><a href="/suspect/{{r.suspect_id}}">{{ r.string_id if r.string_id else r.inverter_id ~ '(整机)' }}</a>{% if r.low_confidence %} <span class="small muted">低置信</span>{% endif %}</td>
<td class="{{ 'neg' if r.dev_today is not none and r.dev_today < -0.05 else '' }}">{{ r.dev_today|pct }}</td><td>{{r.persist_days}}d</td>
<td>{{r.rule_cause}} → <b>{{r.ai_cause or '-'}}</b> ({{ (r.confidence*100)|num(0) if r.confidence is not none else '-' }}%)</td>
<td class="small">{{r.first_date}}</td><td class="small">{{r.action or '-'}}</td>
<td><form class="inline" method="post" action="/review/{{r.suspect_id}}/confirm"><input type="hidden" name="back" value="/review?station={{station}}&grade={{grade}}">
<select name="true_cause"><option value="">按 AI/规则原因</option>{% for c in causes %}<option>{{c}}</option>{% endfor %}</select><button class="ok">确认</button></form>
<form class="inline" method="post" action="/review/{{r.suspect_id}}/reject"><input type="hidden" name="back" value="/review?station={{station}}&grade={{grade}}">
<select name="reason">{% for x in reasons %}<option>{{x}}</option>{% endfor %}</select><button class="no">驳回</button></form></td>
</tr>{% else %}<tr><td colspan=8 class="muted">队列为空</td></tr>{% endfor %}</table></div>
{% endblock %}"""

SUSPECT = r"""{% extends "base.html" %}{% block title %}{{ s.string_id or s.inverter_id }}{% endblock %}{% block body %}
<h2><span class="badge g{{s.grade}}">{{s.grade}}</span> {{ s.string_id or (s.inverter_id ~ '（整机）') }} <span class="muted small">{{ s.suspect_id }} · 状态 {{ s.status }}</span></h2>
<div class="grid g2">
 <div class="card"><h3>全天电流曲线 vs 群组中位</h3>{{ chart_svg|safe }}</div>
 <div class="card"><h3>互比偏差趋势（近 45 天，仅晴好天参与判定）</h3>{{ trend_svg|safe }}</div>
</div>
<div class="grid g2">
<div class="card"><h3>判定</h3>
<p class="expl">{{ s.explanation or '（尚无 AI 判定）' }}</p>
<table><tr><th>规则原因</th><td>{{s.rule_cause}}</td></tr><tr><th>AI 原因</th><td><b>{{s.ai_cause or '-'}}</b> 置信度 {{ (s.confidence*100)|num(0) if s.confidence is not none else '-' }}%（{{ s.model_version or '-' }}）</td></tr>
<tr><th>动作</th><td>{{ s.action or '-' }}</td></tr><tr><th>偏差 / 持续</th><td>{{ s.dev_today|pct }} / {{ s.persist_days }} 天（首报 {{ s.first_date }}，最近 {{ s.date }}）</td></tr>
<tr><th>概率分布</th><td class="small">{% for k,v in proba.items() %}{{k}} {{ (v*100)|num(0) }}%　{% endfor %}</td></tr>
<tr><th>签名</th><td class="small">{% for n in sig.get('notes',[]) %}• {{n}}<br>{% endfor %}
{% if sig.dip_depth %}凹陷 {{ sig.dip_start }}-{{ sig.dip_end }} 深度 {{ (sig.dip_depth*100)|num(0) }}%，其余时段比值 {{ sig.outside_dip_ratio|num(2) }}<br>{% endif %}
{% if sig.overcast_dev is not none %}阴天偏差 {{ sig.overcast_dev|pct }}<br>{% endif %}</td></tr>
<tr><th>损失</th><td>累计 {{ loss.lost_kwh|num(0) }} kWh ≈ {{ loss.lost_yuan|num(0) }} 元（{{ loss.days }} 个晴好天，约 {{ loss.per_day_kwh|num(1) }} kWh/天）</td></tr>
{% if wo %}<tr><th>工单</th><td>{% for w in wo %}<a href="/workorder/{{w.wo_id}}">{{w.wo_id}}</a>（{{w.status}}） {% endfor %}</td></tr>{% endif %}</table>
</div>
<div class="card"><h3>组串档案</h3>
{% if prof %}<table>
<tr><th>逆变器 / MPPT / pv</th><td>{{prof.inverter_id}} / {{prof.mppt}} / pv{{prof.pv_index}}</td></tr>
<tr><th>串长 / 组件</th><td>{{prof.n_modules}} 块 × {{prof.module_model}} {{prof.module_wp}}W（Vmp {{prof.module_vmp}} V，Imp {{prof.module_imp}} A）</td></tr>
<tr><th>朝向 / 倾角</th><td>{{prof.azimuth}}° / {{prof.tilt}}°</td></tr><tr><th>屋面位置</th><td>{{prof.roof_area or '待补'}}</td></tr>
<tr><th>已知遮挡</th><td>{{ '是：' ~ (prof.shading_note or '') if prof.known_shading else '否' }}</td></tr>
<tr><th>互比群组</th><td class="small">{{prof.group_key}}</td></tr><tr><th>备注</th><td>{{prof.notes or ''}}</td></tr></table>
{% else %}<p class="muted">整机级嫌疑（机间互比）</p>{% endif %}
<h3 style="margin-top:14px">人工复核</h3>
{% if s.status in ['open','reviewing','confirmed','dispatched'] %}
<form class="inline" method="post" action="/review/{{s.suspect_id}}/confirm"><input type="hidden" name="back" value="/suspect/{{s.suspect_id}}">
真实原因 <select name="true_cause"><option value="">同 {{ s.ai_cause or s.rule_cause }}</option>{% for c in causes %}<option>{{c}}</option>{% endfor %}</select>
<input type="text" name="note" placeholder="备注"><button class="ok">确认异常</button></form>
<form class="inline" method="post" action="/review/{{s.suspect_id}}/reject" style="margin-top:8px"><input type="hidden" name="back" value="/suspect/{{s.suspect_id}}">
驳回原因 <select name="reason">{% for x in reasons %}<option>{{x}}</option>{% endfor %}</select><input type="text" name="note" placeholder="备注"><button class="no">驳回（误报）</button></form>
{% else %}<p class="muted">该案例已 {{ s.status }}</p>{% endif %}
{% if reviews %}<h3 style="margin-top:14px">复核记录</h3><table>{% for r in reviews %}<tr><td class="small">{{r.reviewed_at[:16]}}</td><td>{{r.reviewer}}</td><td><b>{{r.decision}}</b></td><td>{{r.true_cause or ''}} {{r.reject_reason or ''}}</td><td class="small">{{r.note or ''}}</td></tr>{% endfor %}</table>{% endif %}
</div></div>
{% endblock %}"""

WORKORDERS = r"""{% extends "base.html" %}{% block title %}工单{% endblock %}{% block body %}
<h2>工单 <span class="muted small">draft=草稿待确认发送 · sent=已派 · done=现场已回填待复查 · verified=复查归零 · reopened=未归零重开</span></h2>
<form class="inline card" method="get">状态 <select name="status"><option value="">全部</option>{% for x in ['draft','sent','done','verified','reopened'] %}<option {% if x==status %}selected{% endif %}>{{x}}</option>{% endfor %}</select><button>筛选</button></form>
<div class="card"><table><tr><th>工单</th><th>电站/区域</th><th>原因</th><th>优先级</th><th>损失/天</th><th>累计损失</th><th>状态</th><th>操作</th></tr>
{% for w in rows %}<tr><td><a href="/workorder/{{w.wo_id}}">{{w.wo_id}}</a></td><td>{{w.station_id}} / {{w.roof_area}}</td><td>{{w.cause}}</td><td>{{w.priority|num(1)}}</td>
<td>{{w.loss_kwh_per_day|num(1)}} kWh</td><td>{{w.loss_yuan_total|num(0)}} 元</td><td><b>{{w.status}}</b></td>
<td>{% if w.status in ['draft','reopened'] %}<form class="inline" method="post" action="/workorder/{{w.wo_id}}/send"><button class="pri">确认发送</button></form>{% endif %}</td></tr>
{% else %}<tr><td colspan=8 class="muted">无工单</td></tr>{% endfor %}</table></div>
{% endblock %}"""

WORKORDER = r"""{% extends "base.html" %}{% block title %}{{ wo.wo_id }}{% endblock %}{% block body %}
<h2>{{ wo.wo_id }} <span class="muted small">状态 {{ wo.status }} · 渠道 {{ wo.channel }} · {{ wo.dispatch_result or '' }}</span></h2>
<div class="grid g2"><div class="card"><pre>{{ text }}</pre>
<p class="small">涉及嫌疑：{% for s in suspect_ids %}<a href="/suspect/{{s}}">{{s}}</a> {% endfor %}</p></div>
<div class="card">
{% if wo.status in ['draft','reopened'] %}<h3>派发</h3><form method="post" action="/workorder/{{wo.wo_id}}/send"><button class="pri">确认发送（{{ wo.channel }}）</button></form><hr>{% endif %}
{% if wo.status in ['sent','reopened','draft'] %}<h3>现场结果回填</h3>
<form method="post" action="/workorder/{{wo.wo_id}}/close" class="grid" style="gap:8px">
<label>实际原因 <input type="text" name="field_cause" required placeholder="如：MC4 接头烧蚀"></label>
<label>处理动作 <select name="field_action">{% for a in ['维修','清洗','更换','重接线','换逆变器','无需处理'] %}<option>{{a}}</option>{% endfor %}</select></label>
<label>处理日期 <input type="date" name="done_date"></label>
<label>备注 <textarea name="note" rows="3" placeholder="若为巡检发现但系统未报，请写明“漏报”以统计漏报率"></textarea></label>
<button class="ok">回填并登记事件（基线自该日重置，{{ 3 }} 个晴好天后自动复查）</button></form>
{% elif wo.status=='done' %}<p>已回填：{{ wo.field_cause }} / {{ wo.field_action }} / {{ wo.field_note }}；复查到期 {{ wo.recheck_due }}</p>
{% else %}<p>{{ wo.recheck_result }}</p>{% endif %}
</div></div>
{% endblock %}"""

TEMPLATES = {"base.html": BASE, "index.html": INDEX, "review.html": REVIEW, "suspect.html": SUSPECT,
             "workorders.html": WORKORDERS, "workorder.html": WORKORDER}
