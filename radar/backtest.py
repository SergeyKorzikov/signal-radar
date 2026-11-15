"""Бэктест: поймал бы радар прошлые волны раньше хайпа?

Берём помесячное число статей arXiv по термину (без скачивания — только счётчик)
и нормируем на общий поток arXiv в тех же категориях. Сигнал «загорается», когда
частота за 3 месяца выросла в ≥2 раза при ещё малом абсолютном объёме — то есть
ровно то правило, которым радар ищет ранние сигналы сейчас.
"""
from __future__ import annotations

import calendar
import logging
import math
from datetime import datetime, timezone

from .sources import arxiv

log = logging.getLogger("radar.backtest")
CATS = "cat:cs.LG OR cat:cs.AI OR cat:cs.CL OR cat:cs.CV OR cat:stat.ML"


def _months(a: str, b: str):
    y, m = map(int, a.split("-"))
    y2, m2 = map(int, b.split("-"))
    while (y, m) <= (y2, m2):
        yield y, m
        m += 1
        if m > 12:
            y, m = y + 1, 1


_total_cache: dict[str, int | None] = {}


def series(term_query: str, start: str, end: str) -> list[dict]:
    out = []
    for y, m in _months(start, end):
        a = datetime(y, m, 1, tzinfo=timezone.utc)
        b = datetime(y, m, calendar.monthrange(y, m)[1], tzinfo=timezone.utc)
        n_term = arxiv.count(f"({term_query}) AND ({CATS})", a, b)
        key = f"{y}-{m:02d}"
        if _total_cache.get(key) is None:
            _total_cache[key] = arxiv.count(CATS, a, b)
        n_all = _total_cache[key]
        out.append({"month": f"{y}-{m:02d}", "count": n_term, "total": n_all,
                    "rate": (1e4 * n_term / n_all) if n_term is not None and n_all else None})
    return out


def first_fire(rows: list[dict], ratio: float = 2.0, max_rate: float = 30.0,
               min_count: int = 5) -> str | None:
    """Первый месяц, когда rate(3 мес) / rate(предыдущие 3 мес) ≥ ratio при ещё малом объёме."""
    rates = [r["rate"] for r in rows]
    for i in range(5, len(rows)):
        cur = [x for x in rates[i - 2:i + 1] if x is not None]
        prev = [x for x in rates[i - 5:i - 2] if x is not None]
        if len(cur) < 2 or len(prev) < 2:
            continue
        c, p = sum(cur) / len(cur), sum(prev) / len(prev)
        if rows[i]["count"] and rows[i]["count"] >= min_count and c <= max_rate \
                and math.log2((c + 0.3) / (p + 0.3)) >= math.log2(ratio):
            return rows[i]["month"]
    return None


def months_between(a: str, b: str) -> int:
    y1, m1 = map(int, a.split("-"))
    y2, m2 = map(int, b.split("-"))
    return (y2 - y1) * 12 + (m2 - m1)


def run(cases: list[dict]) -> list[dict]:
    results = []
    for case in cases:
        hype = case["hype"]
        y, m = map(int, hype.split("-"))
        end = f"{y + (m + 2 > 12)}-{(m + 2 - 1) % 12 + 1:02d}"
        log.info("бэктест: %s", case["label"])
        rows = series(case["term"], case["from"], end)
        fire = first_fire(rows)
        lead = months_between(fire, hype) if fire else None
        results.append({"label": case["label"], "hype": hype, "fired": fire,
                        "lead_months": lead, "rows": rows})
    return results
