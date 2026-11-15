"""Hacker News (через Algolia): истории недели и комментарии для «теста игрушки»."""
from __future__ import annotations

import html
import logging
import math
import re
from datetime import datetime, timedelta, timezone

from ..http import get
from ..models import Doc, Window

log = logging.getLogger("radar.hn")
API = "https://hn.algolia.com/api/v1/search_by_date"

# Скептики: «это игрушка». Энтузиасты: «я не сплю третью ночь».
SKEPTIC = re.compile(
    r"\b(toy|gimmick|useless|overhyped|hype|snake oil|doesn'?t work|won'?t work|"
    r"solution in search of|nobody needs|fad|vaporware|parlor trick|just a wrapper|"
    r"not useful|pointless|meh|nothing new)\b", re.I)
ENTHUSIAST = re.compile(
    r"\b(game[- ]?changer|mind[- ]?blow\w*|blown away|insane|incredible|obsessed|"
    r"can'?t stop|addicted|magic(al)?|holy|wow|this changes|future of|"
    r"been playing with|spent all (night|weekend)|finally)\b", re.I)


def fetch(window: Window, min_points: int = 5) -> list[Doc]:
    docs: list[Doc] = []
    a, b = int(window.start.timestamp()), int(window.end.timestamp())
    # Algolia отдаёт до 1000 результатов на запрос — идём по дням
    t = a
    while t < b:
        t2 = min(t + 86400, b)
        page = 0
        while True:
            r = get(API, params={"tags": "story", "hitsPerPage": 1000, "page": page,
                                 "numericFilters": f"created_at_i>={t},created_at_i<{t2},"
                                                   f"points>={min_points}"}, host_delay=0.2)
            if r is None:
                break
            data = r.json()
            for h in data.get("hits", []):
                title = h.get("title") or ""
                pts = h.get("points") or 0
                show = title.lower().startswith(("show hn", "launch hn"))
                docs.append(Doc(
                    source="hn", id=str(h.get("objectID")), title=title,
                    text=f"{title}. {html.unescape(re.sub(r'<[^>]+>', ' ', h.get('story_text') or ''))[:600]}",
                    url=f"https://news.ycombinator.com/item?id={h.get('objectID')}",
                    weight=1.0 + math.log1p(pts) / 2,
                    domain="show" if show else "story",
                    meta={"points": pts, "comments": h.get("num_comments") or 0,
                          "link": h.get("url") or ""}))
            page += 1
            if page >= (data.get("nbPages") or 0):
                break
        t = t2
    log.info("HN %s: %d историй", window.week_id, len(docs))
    return docs


def toy_test(term: str, days: int = 30, now: datetime | None = None,
             sample: int = 300) -> dict:
    """Ищем в комментариях одновременно скептиков и энтузиастов."""
    now = now or datetime.now(timezone.utc)
    since = int((now - timedelta(days=days)).timestamp())
    r = get(API, params={"tags": "comment", "query": f'"{term}"', "hitsPerPage": sample,
                         "numericFilters": f"created_at_i>={since}"}, host_delay=0.3)
    if r is None:
        return {"comments": 0, "skeptic": 0, "enthusiast": 0, "examples": []}
    s = e = 0
    examples = []
    for h in r.json().get("hits", []):
        txt = html.unescape(re.sub(r"<[^>]+>", " ", h.get("comment_text") or ""))
        if term.lower() not in txt.lower():
            continue
        is_s, is_e = bool(SKEPTIC.search(txt)), bool(ENTHUSIAST.search(txt))
        s += is_s
        e += is_e
        if (is_s or is_e) and len(examples) < 4:
            m = (SKEPTIC if is_s else ENTHUSIAST).search(txt)
            lo = max(0, m.start() - 110)
            snippet = " ".join(txt[lo:m.end() + 110].split())
            examples.append({"side": "скептик" if is_s else "энтузиаст", "text": snippet,
                             "url": f"https://news.ycombinator.com/item?id={h.get('objectID')}"})
    n = r.json().get("nbHits", 0)
    return {"comments": n, "skeptic": s, "enthusiast": e, "examples": examples}
