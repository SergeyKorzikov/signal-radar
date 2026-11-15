"""Полный цикл: сбор → частоты → оценка → нейросеть-фильтр → занятость → разбор → подборка."""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import analyst, history, lanes, occupancy, scoring
from .models import Window
from .sources import arxiv, biorxiv, gdelt, github, hn, huggingface
from .terms import WeekIndex

log = logging.getLogger("radar")
ROOT = Path(__file__).resolve().parent.parent
SCIENCE, TECH = "наука", "технологии"
LANES = [SCIENCE, TECH, lanes.CULTURE, lanes.GAMES, lanes.MARKETS]


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


def _save_week(w: Window, idx: WeekIndex, items: list) -> None:
    history.save_week(w.week_id, {"week": w.week_id, "start": w.start.isoformat(),
                                  "end": w.end.isoformat(), **idx.to_history(),
                                  "entities": lanes.to_history(items)})


def backfill(weeks: int, cfg: dict, end: Window | None = None) -> None:
    cur = end or Window.last_full_week()
    existing = set(history.list_weeks())
    for k in range(weeks, 0, -1):
        w = cur.previous(k)
        if w.week_id in existing:
            log.info("неделя %s уже есть", w.week_id)
            continue
        log.info("=== догружаю историю: %s ===", w.week_id)
        _save_week(w, WeekIndex(collect(w, cfg, live=False)), lanes.collect(w, cfg, live=False))


def _query(c: dict) -> str:
    """Короткий поисковый запрос из названия сигнала (для проверки занятости)."""
    q = re.sub(r"\[[^\]]*\]|\([^)]*\)|[^\w\s'-]", " ", c["label"])
    return " ".join(q.split()[:4])


def choose(pool: list[dict], cfg: dict) -> tuple[list[dict], list[dict]]:
    """Лучшие сигналы каждой дорожки + принудительные слоты «вне пузыря»."""
    sel, bonus = cfg["selection"], cfg["scoring"]["fit_bonus"]
    alive = [c for c in pool if c.get("keep", True)]
    for c in alive:
        c["rank"] = c["score"] * (1 + bonus * c.get("fit", 0))
    main: list[dict] = []
    for lane in LANES:
        main += sorted((c for c in alive if c["lane"] == lane),
                       key=lambda c: -c["rank"])[:sel["per_lane"]]
    chosen = {c["term"] for c in main}
    outside = [c for c in sorted(alive, key=lambda c: -c["score"])
               if c["term"] not in chosen and c.get("fit", 0) < 0.15][:sel["outside_bubble"]]
    return main, outside


def run_analyst(pool: list[dict], cfg: dict, use_llm: bool = True):
    """Фильтр → выбор → занятость → разбор. backend: local (своя нейросеть) или none."""
    a = cfg.get("analyst", {})
    backend = a.get("backend", "local") if use_llm else "none"
    llm = None
    if backend == "local":
        try:
            from .local_llm import LocalAnalyst
            llm = LocalAnalyst(a["local"])
        except Exception as e:  # noqa: BLE001 — без модели радар всё равно должен отработать
            log.warning("локальная модель недоступна (%s) — разборы будут шаблонными", e)
    if llm:
        llm.filter(pool)
    main, outside = choose(pool, cfg)
    if use_llm:
        for c in main + outside:
            if c["lane"] == lanes.MARKETS:
                continue  # для рынка предсказаний «сколько игр в Roblox» не имеет смысла
            c["occupancy"] = occupancy.check(_query(c))
            c["occupancy_text"] = occupancy.describe(c["occupancy"])
    if llm:
        from .local_llm import apply_write
        for c in main + outside:
            apply_write(c, llm.write(c))
        return main, outside, llm.summarize(main), "local"
    for c in main + outside:
        c.update(analyst.template_writeup(c))
    return main, outside, None, "template"


def analyze(idx: WeekIndex, window: Window, hist: list[dict], cfg: dict, tracker: dict,
            items: list | None = None, enrich_remote: bool = True) -> dict:
    sc = cfg["scoring"]
    # ---- наука и технологии: рост терминов
    cands, wave = scoring.build_candidates(idx, hist, cfg, tracker)
    log.info("наука/технологии: кандидатов %d, «уже волна» %d", len(cands), len(wave))
    scoring.score(cands, cfg)
    top = sorted(cands, key=lambda c: c["score"], reverse=True)[:400]
    top = scoring.dedupe(top, idx.postings)[:300]
    for c in top:
        scoring.enrich_static(c, idx, cfg["interests"], tracker)
        bk = c["by_kind"]
        c["lane"] = SCIENCE if bk["research"] >= bk["builder"] + bk["talk"] else TECH
    scoring.score(top, cfg)
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

    # ---- быстрые дорожки: культура и мемы, игры, рынки
    lane_cands = lanes.build_candidates(items or [], hist, cfg, tracker, cfg["interests"])
    lane_pool = lanes.pick_pool(lane_cands, cfg.get("lanes", {}).get("pool_per_lane", 10))
    log.info("быстрые дорожки: кандидатов %d, в пуле %d", len(lane_cands), len(lane_pool))
    if enrich_remote:
        for c in lane_pool:  # нейросети нужно описание: для статей Википедии берём аннотацию
            for e in [e for e in c["examples"] if e["source"] == "wikipedia"][:2]:
                if not e.get("snippet"):
                    e["snippet"] = lanes.wiki_summary(e["url"].rsplit("/", 1)[-1])

    pool = sorted(pool, key=lambda c: c["score"], reverse=True) + lane_pool
    main, outside, summary, mode = run_analyst(pool, cfg, use_llm=enrich_remote)
    n_wave = cfg["selection"]["already_wave"]

    def slim(c: dict) -> dict:
        return {k: v for k, v in c.items() if k not in ("rank",)}

    return {
        "week": window.week_id, "start": window.start.isoformat(), "end": window.end.isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "analyst": mode, "summary": summary, "lanes": LANES,
        "stats": {"docs": dict(idx.n_docs), "candidates": len(cands) + len(lane_cands),
                  "history_weeks": len(hist),
                  "items": {s: len(v) for s, v in lanes.to_history(items or []).items()}},
        "main": [slim(c) for c in main], "outside": [slim(c) for c in outside],
        "wave": [slim(c) for c in wave[:n_wave]],
        "rejected": [{"term": c["label"], "lane": c.get("lane"), "reason": c.get("reject_reason")}
                     for c in pool if c.get("keep") is False],
    }


def update_registry(digest: dict, path: Path | None = None) -> dict:
    p = path or ROOT / "data" / "signals.json"
    reg = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    stamp = digest["generated_at"][:10]
    for section in ("main", "outside"):
        for c in digest[section]:
            r = reg.setdefault(c["term"], {"label": c["label"], "first_week": digest["week"],
                                           "first_seen": stamp, "weeks": [], "scores": []})
            r["lane"] = c.get("lane")
            if digest["week"] not in r["weeks"]:
                r["weeks"].append(digest["week"])
                r["scores"].append(c["score"])
            else:
                r["scores"][-1] = c["score"]
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
    items = lanes.collect(window, cfg)
    _save_week(window, idx, items)
    hist = [history.load_week(w) for w in history.list_weeks() if w < window.week_id]
    for h in hist:  # недели, собранные до появления быстрых дорожек: догружаем Википедию
        if "entities" not in h and h.get("start"):
            hw = Window(datetime.fromisoformat(h["start"]), datetime.fromisoformat(h["end"]))
            h["entities"] = lanes.to_history(lanes.collect(hw, cfg, live=False))
            history.save_week(h["week"], h)
    digest = analyze(idx, window, hist, cfg, load_tracker(), items)
    history.save_digest(window.week_id, digest)
    reg = update_registry(digest)
    history.compact(set(reg))
    return digest
