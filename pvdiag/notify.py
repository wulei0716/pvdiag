"""关键告警推送：系统自己能处理的事不打扰人，处理不了的（管道断了 / 门控收回 / 模型降级 / 参数回滚）一定要让人知道。

渠道：企业微信群机器人 webhook（alerts.wecom_webhook 或 dispatch.wecom_webhook）；没配就写 data/outbox/alerts/<date>.md。
去重：同一内容 24h 内只推一次（platform_state 记指纹），避免每日巡检重复刷屏。
"""
from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional

from .config import Settings
from .storage import Store

LEVELS = {"info": 0, "warn": 1, "error": 2}


class Notifier:
    def __init__(self, settings: Settings, store: Store):
        self.settings = settings
        self.store = store
        self.cfg = settings.alerts
        self.webhook = self.cfg.wecom_webhook or settings.dispatch.wecom_webhook

    def _fingerprint(self, text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    def _recently_sent(self, fp: str) -> bool:
        rec = self.store.get_state(f"notify:sent:{fp}")
        if not rec:
            return False
        try:
            return datetime.fromisoformat(rec["at"]) > datetime.now() - timedelta(hours=self.cfg.dedupe_hours)
        except Exception:
            return False

    def send(self, run_date: date, title: str, lines: List[str], level: str = "error", force: bool = False) -> Optional[str]:
        """返回投递结果描述；被去重/低于门槛/关闭 → None"""
        if not self.cfg.enabled or not lines:
            return None
        if not force and LEVELS.get(level, 0) < LEVELS.get(self.cfg.min_level, 2):
            return None
        body = f"## pvdiag {title}（{run_date.isoformat()}）\n" + "\n".join(f"- {l}" for l in lines)
        fp = self._fingerprint("\n".join(sorted(lines)))
        if self._recently_sent(fp):
            return None
        result = self._deliver(run_date, body)
        self.store.set_state(f"notify:sent:{fp}", {"at": datetime.now().isoformat(), "title": title, "result": result})
        self.store.execute("INSERT INTO alerts (run_date, level, source, message, created_at) VALUES (?,?,?,?,?)",
                           (run_date.isoformat(), "info", "notify", f"已推送「{title}」→ {result}", datetime.now().isoformat()))
        return result

    def _deliver(self, run_date: date, body: str) -> str:
        if self.webhook:
            try:
                import requests
                r = requests.post(self.webhook, json={"msgtype": "markdown", "markdown": {"content": body[:4000]}}, timeout=20)
                r.raise_for_status()
                if r.json().get("errcode") not in (0, None):
                    raise RuntimeError(str(r.json()))
                return "wecom:ok"
            except Exception as e:      # 推送失败也要落地，别丢
                self._write_outbox(run_date, body + f"\n\n> 企业微信推送失败：{e}")
                return f"wecom:failed→outbox ({e})"
        return self._write_outbox(run_date, body)

    def _write_outbox(self, run_date: date, body: str) -> str:
        folder: Path = self.settings.outbox_dir / "alerts"
        folder.mkdir(parents=True, exist_ok=True)
        p = folder / f"{run_date.isoformat()}.md"
        with open(p, "a", encoding="utf-8") as f:
            f.write(body + "\n\n")
        return f"outbox:{p}"
