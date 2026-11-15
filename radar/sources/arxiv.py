"""arXiv: свежие препринты по выбранным категориям."""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from datetime import datetime

from ..http import get
from ..models import Doc, Window

log = logging.getLogger("radar.arxiv")
API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom",
      "os": "http://a9.com/-/spec/opensearch/1.1/"}
DELAY = 4.0  # правила arXiv: не чаще раза в 3 секунды (берём с запасом)


def _stamp(d: datetime, end: bool = False) -> str:
    return d.strftime("%Y%m%d") + ("2359" if end else "0000")


def date_clause(start: datetime, end_inclusive: datetime) -> str:
    return f"submittedDate:[{_stamp(start)} TO {_stamp(end_inclusive, True)}]"


def count(query: str, start: datetime, end_inclusive: datetime) -> int | None:
    """Сколько статей отвечают запросу за период (для бэктеста, без скачивания)."""
    q = f"({query}) AND {date_clause(start, end_inclusive)}"
    r = get(API, params={"search_query": q, "max_results": 1}, host_delay=DELAY, retries=7)
    if r is None:
        return None
    try:
        root = ET.fromstring(r.content)
        return int(root.find("os:totalResults", NS).text)
    except Exception:  # noqa: BLE001
        return None


def fetch(window: Window, categories: list[str], max_docs: int = 12000,
          page: int = 500) -> list[Doc]:
    from datetime import timedelta

    cats = " OR ".join(f"cat:{c}" for c in categories)
    q = f"({cats}) AND {date_clause(window.start, window.end - timedelta(seconds=1))}"
    docs: list[Doc] = []
    start = 0
    total = None
    while start < max_docs:
        r = get(API, params={"search_query": q, "start": start, "max_results": page,
                             "sortBy": "submittedDate", "sortOrder": "ascending"},
                host_delay=DELAY, timeout=120, retries=7)
        if r is None:
            break
        root = ET.fromstring(r.content)
        if total is None:
            total = int(root.find("os:totalResults", NS).text)
            log.info("arXiv %s: %d статей", window.week_id, total)
        entries = root.findall("a:entry", NS)
        if not entries:
            break
        for e in entries:
            title = " ".join((e.findtext("a:title", "", NS) or "").split())
            summary = " ".join((e.findtext("a:summary", "", NS) or "").split())
            prim = e.find("arxiv:primary_category", NS)
            domain = prim.get("term") if prim is not None else ""
            url = (e.findtext("a:id", "", NS) or "").replace("http://", "https://")
            docs.append(Doc(source="arxiv", id=url.rsplit("/", 1)[-1], title=title,
                            text=f"{title}. {summary}", url=url, weight=1.0, domain=domain,
                            meta={"published": e.findtext("a:published", "", NS)}))
        start += len(entries)
        if total is not None and start >= total:
            break
    return docs
