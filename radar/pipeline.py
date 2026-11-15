"""Полный недельный цикл: сбор → частоты → оценка → проверка → разбор → подборка."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import analyst, history, scoring
from .models import Window
from .sources import arxiv, biorxiv, gdelt, github, hn, huggingface
from .terms import WeekIndex

log = logging.getLogger("radar")
ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path | None = None) -> dict:
    return yaml.safe_load((path or ROOT / "config.yaml").read_text(encoding="utf-8"))


def load_tracker(path: Path | None = None) -> dict:
    p = path or ROOT / "tracker.yaml"
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return {str(k).lower(): v for k, v in (data.get("signals") or {}).items()}


def collect(window: Window, cfg: dict, live: bool = True) -> list:
    """live=False — для исторических недель (у Hugging Face нет прошлых срезов)."""
    s = cfg["sources"]
    docs = []
    if s["arxiv"]["enabled"]:
        docs += arxiv.fetch(window, s["arxiv"]["categories"], s["arxiv"].get("max_docs", 15000))
    if s["biorxiv"]["enabled"]:
        docs += biorxiv.fetch(window, s["biorxiv"]["servers"], s["biorxiv"].get("categories") or None)
    if s["github"]["enabled"]:
        docs += github.fetch(window, s["github"].get("min_stars", 10))
    if s["huggingface"]["enabled"] and live:
        docs += huggingface.fetch(window)
    if s["hn"]["enabled"]:
        docs += hn.fetch(window, s["hn"].get("min_points", 5))
    return docs


def backfill(weeks: int, cfg: dict, end: Window | None = None) -> None:
    cur = end or Window.last_full_week()
    existing = set(history.list_weeks())
    for k in range(weeks, 0, -1):
        w = cur.previous(k)
        if w.week_id in existing:
            log.info("неделя %s уже есть", w.week_id)
            continue
        log.info("=== догружаю историю: %s ===", w.week_id)
        idx = WeekIndex(collect(w, cfg, live=False))
        history.save_week(w.week_id, {"week": w.week_id, "start": w.start.isoformat(),
                                      "end": w.end.isoformat(), **idx.to_history()})


def analyze(idx: WeekIndex, window: Window, hist: list[dict], cfg: dict, tracker: dict,
            enrich_remote: bool = True) -> dict:
    sc = cfg["scoring"]
    cands, wave = scoring.build_candidates(idx, hist, cfg, tracker)
    log.info("кандидатов: %d, «уже волна»: %d", len(cands), len(wave))
    scoring.score(cands, cfg)
    top = sorted(cands, key=lambda c: c["score"], reverse=True)[:400]
    top = scoring.dedupe(top, idx.postings)[:300]
    for c in top:
        scoring.enrich_static(c, idx, cfg["interests"], tracker)
    scoring.score(top, cfg)

    # пул для глубокой проверки: лучшие с учётом ваших интересов + лучшие «чужие»
    n_pool = sc["candidate_pool"]
    for c in top:
        c["rank"] = c["score"] * (1 + sc["fit_bonus"] * c["fit"])
    pool = sorted(top, key=lambda c: c["rank"], reverse=True)[:int(n_pool * 0.7)]
    names = {c["term"] for c in pool}
    pool += [c for c in sorted(top, key=lambda c: c["score"], reverse=True)
             if c["fit"] < 0.15 and c["term"] not in names][:n_pool - len(pool)]

    if enrich_remote:
        gdelt_on, gdelt_fails = cfg["sources"]["gdelt"]["enabled"], 0
        for c in pool:
            q = c["label"]
            c["toy"] = hn.toy_test(q)
            c["toy_score"] = scoring.toy_score(c["toy"])
            if gdelt_on and len(q) >= 5:
                c["media"] = gdelt.media_count(q)
                gdelt_fails = gdelt_fails + 1 if c["media"] is None else 0
                if gdelt_fails >= 3:  # GDELT лежит — не тратим на него время
                    log.warning("GDELT не отвечает — медиа-фон на этой неделе нейтральный")
                    gdelt_on = False
            c["media_gap"] = scoring.media_gap(c.get("media"), c["now"])
        scoring.score(top, cfg)

    pool.sort(key=lambda c: c["score"], reverse=True)
    main, outside, summary, mode = run_analyst(pool, cfg, use_llm=enrich_remote)
    for c in wave[:cfg["selection"]["already_wave"]]:
        c.setdefault("label", c["term"])

    def slim(c: dict) -> dict:
        return {k: v for k, v in c.items() if k not in ("rank",)}

    return {
        "week": window.week_id, "start": window.start.isoformat(), "end": window.end.isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "analyst": mode, "summary": summary,
        "stats": {"docs": dict(idx.n_docs), "candidates": len(cands), "history_weeks": len(hist)},
        "main": [slim(c) for c in main], "outside": [slim(c) for c in outside],
        "wave": [slim(c) for c in wave[:cfg["selection"]["already_wave"]]],
        "rejected": [{"term": c["label"], "reason": c.get("reject_reason")}
                     for c in pool if c.get("keep") is False],
    }


def run_analyst(pool: list[dict], cfg: dict, use_llm: bool = True):
    """Фильтр + разбор. backend: local (своя нейросеть), claude (API) или none (шаблоны)."""
    a = cfg.get("analyst", {})
    backend = a.get("backend", "local") if use_llm else "none"
    sc = cfg["scoring"]
    if backend == "local":
        try:
            from .local_llm import LocalAnalyst, apply_write
            llm = LocalAnalyst(a["local"])
        except Exception as e:  # noqa: BLE001 — без модели радар всё равно должен отработать
            log.warning("локальная модель недоступна (%s) — разборы будут шаблонными", e)
            backend = "none"
        else:
            llm.filter(pool)
            main, outside = scoring.select(pool, cfg, sc["fit_bonus"])
            for c in main + outside:
                apply_write(c, llm.write(c, cfg["profile"]))
            return main, outside, llm.summarize(main), "local"
    if backend == "claude":
        review = analyst.review(pool, cfg["profile"], a.get("claude", {}))
        summary = analyst.apply_review(pool, review)
        main, outside = scoring.select(pool, cfg, sc["fit_bonus"])
        return main, outside, summary, "ai" if review else "template"
    analyst.apply_review(pool, None)
    main, outside = scoring.select(pool, cfg, sc["fit_bonus"])
    return main, outside, None, "template"


def update_registry(digest: dict, path: Path | None = None) -> dict:
    p = path or ROOT / "data" / "signals.json"
    reg = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    for section in ("main", "outside"):
        for c in digest[section]:
            r = reg.setdefault(c["term"], {"label": c["label"], "first_week": digest["week"],
                                           "weeks": [], "scores": []})
            if digest["week"] not in r["weeks"]:
                r["weeks"].append(digest["week"])
                r["scores"].append(c["score"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(reg, ensure_ascii=False, indent=1), encoding="utf-8")
    return reg


def run(cfg: dict, window: Window | None = None, auto_backfill: int = 6) -> dict:
    window = window or Window.last_full_week()
    if len([w for w in history.list_weeks() if w < window.week_id]) < 3 and auto_backfill:
        log.info("истории мало — сначала догружаю %d недель", auto_backfill)
        backfill(auto_backfill, cfg, window)
    log.info("=== неделя %s ===", window.week_id)
    idx = WeekIndex(collect(window, cfg))
    history.save_week(window.week_id, {"week": window.week_id, "start": window.start.isoformat(),
                                       "end": window.end.isoformat(), **idx.to_history()})
    hist = [history.load_week(w) for w in history.list_weeks() if w < window.week_id]
    digest = analyze(idx, window, hist, cfg, load_tracker())
    history.save_digest(window.week_id, digest)
    reg = update_registry(digest)
    history.compact(set(reg))
    return digest
