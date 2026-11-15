"""Проверки алгоритма на синтетических данных (без сети)."""
from datetime import datetime, timezone

import yaml

from radar import analyst, pipeline, scoring
from radar.models import Doc, Window
from radar.terms import WeekIndex, extract

CFG = yaml.safe_load(open(pipeline.ROOT / "config.yaml", encoding="utf-8"))


def test_extract_skips_boilerplate_and_normalizes_plurals():
    t = extract("We propose a novel framework. Spiking world models enable fast recall.")
    assert "spiking world model" in t
    assert "world model" in t
    assert not any(k.startswith("propose") or k.endswith("framework") for k in t)


def _doc(src, i, text, w=1.0, domain=""):
    return Doc(source=src, id=f"{src}{i}", title=text[:60], text=text, url="https://x", weight=w,
               domain=domain)


def _week(n_growing: int, i0: int) -> list[Doc]:
    docs = []
    for i in range(300):  # фон
        docs.append(_doc("arxiv", i0 + i, f"large language model evaluation on topic {i % 17} graph"))
    for i in range(60):
        docs.append(_doc("github", i0 + 500 + i, f"cli utility number {i % 7} for kubernetes"))
        docs.append(_doc("hn", i0 + 600 + i, f"Ask HN: career advice about remote work {i % 5}"))
    for i in range(n_growing):
        docs.append(_doc("arxiv", i0 + 1000 + i, "Neural dust implants for chronic cortex recording"))
        docs.append(_doc("github", i0 + 2000 + i, "neural dust implant firmware toolkit", w=3.0))
        docs.append(_doc("hn", i0 + 3000 + i, "Show HN: neural dust implant simulator"))
    for i in range(11 if n_growing >= 9 else 0):  # новое из далёкой области (анти-пузырь)
        docs.append(_doc("arxiv", i0 + 4000 + i, f"Perovskite tandem photovoltaic stability {i}",
                         domain="cond-mat.mtrl-sci"))
    return docs


def _history():
    hist = []
    for k, n in enumerate([0, 0, 0, 1, 0, 1]):
        idx = WeekIndex(_week(n, k * 10000))
        hist.append({"week": f"2026-W{30 + k}", **idx.to_history(min_df=1)})
    return hist


def test_growing_multi_source_term_ranks_first_and_mainstream_is_excluded():
    hist = _history()
    cur = WeekIndex(_week(9, 99000))
    cands, wave = scoring.build_candidates(cur, hist, CFG, {})
    scoring.score(cands, CFG)
    top = sorted(cands, key=lambda c: c["score"], reverse=True)
    top = scoring.dedupe(top, cur.postings)
    assert "neural dust implant" in [c["term"] for c in top[:3]]
    names = {c["term"] for c in cands}
    assert "large language model" not in names  # стабильный фон — не сигнал
    best = next(c for c in top if c["term"] == "neural dust implant")
    assert best["n_kinds"] == 3 and best["growth"] > 1


def test_selection_reserves_outside_bubble_slots():
    hist = _history()
    cur = WeekIndex(_week(9, 99000))
    w = Window(datetime(2026, 11, 8, tzinfo=timezone.utc), datetime(2026, 11, 15, tzinfo=timezone.utc))
    d = pipeline.analyze(cur, w, hist, CFG, {}, enrich_remote=False)
    assert d["main"], "основная подборка пуста"
    assert all(c["fit"] < 0.15 for c in d["outside"])
    assert any("perovskite" in c["term"] for c in d["outside"] + d["main"])
    for c in d["main"]:
        assert c["analysis"]["cheap_bet"] and 0 <= c["earn"] <= 10 and 0 <= c["learn"] <= 10


def test_muted_terms_are_hidden():
    hist = _history()
    cur = WeekIndex(_week(9, 99000))
    cands, _ = scoring.build_candidates(cur, hist, CFG, {"neural dust implant": {"status": "мёртв"}})
    assert "neural dust implant" not in {c["term"] for c in cands}


def test_analyst_review_rejects_and_rewrites():
    c = {"term": "x y", "label": "x y", "by_source": {"arxiv": 5}, "by_kind": {"research": 5, "builder": 0,
         "talk": 0}, "growth": 2, "expected": 1, "now": 5}
    d = dict(c, term="junk")
    summary = analyst.apply_review([c, d], {"signals": [
        {"term": "x y", "keep": True, "title_ru": "Икс", "cheap_bet": "b", "earn": 7, "learn": 3},
        {"term": "junk", "keep": False, "reject_reason": "общая фраза"}], "week_summary": "s"})
    assert summary == "s" and c["earn"] == 7 and c["analysis"]["title_ru"] == "Икс"
    assert d["keep"] is False


def test_toy_and_media_scores():
    assert scoring.toy_score({"skeptic": 3, "enthusiast": 3}) > scoring.toy_score({"enthusiast": 3})
    assert scoring.media_gap(0, 10) == 1.0
    assert scoring.media_gap(None, 10) == 0.5
    assert scoring.media_gap(5000, 10) < 0.1


def test_noise_filters():
    hist = _history()
    docs = _week(9, 99000)
    for i in range(12):  # новостной всплеск только на HN
        docs.append(_doc("hn", 500000 + i, "Nvidia CEO Jensen Huang interview"))
    for i in range(10):  # обычное слово, встречающееся вскользь
        docs.append(_doc("arxiv", 600000 + i, f"Paper {i}. A long journey through optimization."))
    cur = WeekIndex(docs)
    names = {c["term"] for c in scoring.build_candidates(cur, hist, CFG, {})[0]}
    assert "jensen huang" not in names and "journey" not in names
    assert "neural dust implant" in names
