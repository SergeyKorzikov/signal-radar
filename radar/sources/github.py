"""GitHub: новые репозитории, которые быстро набирают звёзды."""
from __future__ import annotations

import logging
import math
import os
from datetime import datetime, timedelta, timezone

from ..http import get
from ..models import Doc, Window

log = logging.getLogger("radar.github")
API = "https://api.github.com/search/repositories"


def _headers() -> dict:
    h = {"Accept": "application/vnd.github+json"}
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def _search(q: str, max_pages: int) -> list[dict]:
    items: list[dict] = []
    for page in range(1, max_pages + 1):
        r = get(API, params={"q": q, "sort": "stars", "order": "desc", "per_page": 100,
                             "page": page}, headers=_headers(), host_delay=2.5)
        if r is None:
            break
        batch = r.json().get("items", [])
        items.extend(batch)
        if len(batch) < 100:
            break
    return items


def fetch(window: Window, min_stars: int = 10, max_pages: int = 10,
          now: datetime | None = None) -> list[Doc]:
    """Репозитории, созданные в окне. Поиск GitHub отдаёт до 1000 результатов на запрос,
    поэтому неделю делим на дни."""
    now = now or datetime.now(timezone.utc)
    seen: set[int] = set()
    docs: list[Doc] = []
    day = window.start
    while day < window.end:
        nxt = day + timedelta(days=1)
        q = f"created:{day:%Y-%m-%d} stars:>={min_stars}"
        for it in _search(q, max_pages=max(1, max_pages // 2)):
            if it["id"] in seen or it.get("fork"):
                continue
            seen.add(it["id"])
            created = datetime.fromisoformat(it["created_at"].replace("Z", "+00:00"))
            age_days = max((now - created).total_seconds() / 86400, 1.0)
            stars = it.get("stargazers_count", 0)
            velocity = stars / age_days  # звёзд в сутки: честнее, чем просто звёзды
            topics = it.get("topics") or []
            name = it.get("name", "")
            desc = it.get("description") or ""
            docs.append(Doc(
                source="github", id=it["full_name"], title=f"{it['full_name']}: {desc}"[:300],
                text=f"{name.replace('-', ' ').replace('_', ' ')}. {desc}. {' '.join(topics)}",
                url=it["html_url"], weight=1.0 + math.log1p(velocity),
                domain=(it.get("language") or ""), tags=[t.replace("-", " ") for t in topics],
                meta={"stars": stars, "stars_per_day": round(velocity, 1)}))
        day = nxt
    log.info("GitHub %s: %d репозиториев", window.week_id, len(docs))
    return docs
