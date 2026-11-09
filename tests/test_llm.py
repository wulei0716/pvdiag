"""LLM 助手：案卷组装、模板回退、调用缓存、追问记录、控制台入口（大模型调用用 monkeypatch 假客户端）"""
from datetime import timedelta

import pytest

from pvdiag import llm as llm_mod
from pvdiag.orchestrator import DailyPipeline


def _case(sim_env):
    settings, store, data, cfg = sim_env
    end = cfg.start + timedelta(days=cfg.days - 1)
    DailyPipeline(settings, store=store, ingest=False).backfill(end - timedelta(days=12), end)
    sid = store.scalar("SELECT suspect_id FROM suspects WHERE grade IN ('A','B') ORDER BY grade LIMIT 1")
    return settings, store, sid


def test_dossier_and_template_fallback(sim_env, monkeypatch):
    settings, store, sid = _case(sim_env)
    monkeypatch.delenv(settings.llm.api_key_env, raising=False)
    d = llm_mod.case_dossier(store, sid)
    assert d["case"]["suspect_id"] == sid and "signature" in d and "recent_days" in d
    r = llm_mod.narrative(settings, store, sid)
    assert r["source"] == "template" and "【结论】" in r["text"] and "【建议】" in r["text"]
    a = llm_mod.ask(settings, store, sid, "损失多少")
    assert a["source"] == "template" and "kWh" in a["text"]
    assert store.scalar("SELECT COUNT(*) FROM reviews WHERE suspect_id=? AND decision='note'", (sid,)) == 1
    # 追问记录不污染标注统计
    from pvdiag.agents.review import ReviewAgent
    assert ReviewAgent.stats(store, days=365)["n_reviewed"] == 0


def test_llm_called_once_and_cached(sim_env, monkeypatch):
    settings, store, sid = _case(sim_env)
    monkeypatch.setenv(settings.llm.api_key_env, "test-key")
    calls = []

    def fake_complete(self, system, user):
        calls.append(user)
        assert "案卷" in user and sid in user
        return "【结论】测试分析\n【依据】x\n【建议】y\n【不确定性】z"
    monkeypatch.setattr(llm_mod.LLMClient, "complete", fake_complete)
    r1 = llm_mod.narrative(settings, store, sid)
    r2 = llm_mod.narrative(settings, store, sid)
    assert r1["source"].startswith("llm:") and r1["text"].startswith("【结论】") and len(calls) == 1 and r2["text"] == r1["text"]
    # 案卷变化（新增复核）→ 指纹变 → 重新生成
    from pvdiag.agents.review import ReviewAgent
    ReviewAgent.confirm(store, sid, None, "t")
    llm_mod.narrative(settings, store, sid)
    assert len(calls) == 2
    # 调用失败 → 回退模板并标注失败原因
    def boom(self, system, user):
        raise RuntimeError("429 rate limited")
    monkeypatch.setattr(llm_mod.LLMClient, "complete", boom)
    a = llm_mod.ask(settings, store, sid, "为什么这样判")
    assert a["source"] == "template" and "429" in a["text"]
    # 关闭后不调用
    settings.llm.enabled = False
    assert not llm_mod.LLMClient(settings).available
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from pvdiag.web.app import create_app
    c = TestClient(create_app(settings))
    assert "分析助手" in c.get(f"/suspect/{sid}").text
    r = c.post(f"/suspect/{sid}/ask", data={"question": "现场怎么办"}, follow_redirects=False)
    assert r.status_code == 303
