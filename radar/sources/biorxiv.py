"""bioRxiv / medRxiv: препринты по биологии и медицине (нейронауки в том числе)."""
from __future__ import annotations

import logging
from datetime import timedelta

from ..http import get
from ..models import Doc, Window

log = logging.getLogger("radar.biorxiv")


def fetch(window: Window, servers: list[str], categories: list[str] | None = None,
          max_docs: int = 6000) -> list[Doc]:
    cats = {c.lower() for c in categories} if categories else None
    a = window.start.strftime("%Y-%m-%d")
    b = (window.end - timedelta(days=1)).strftime("%Y-%m-%d")
    docs: list[Doc] = []
    for server in servers:
        cursor = 0
        while cursor < max_docs:
            r = get(f"https://api.biorxiv.org/details/{server}/{a}/{b}/{cursor}", host_delay=0.5)
            if r is None:
                break
            data = r.json()
            coll = data.get("collection") or []
            if not coll:
                break
            for p in coll:
                if str(p.get("version", "1")) != "1":
                    continue  # только первые версии, иначе одна статья считается много раз
                cat = (p.get("category") or "").strip().lower()
                if cats and cat not in cats:
                    continue
                title = " ".join((p.get("title") or "").split())
                abstract = " ".join((p.get("abstract") or "").split())
                doi = p.get("doi", "")
                docs.append(Doc(source="biorxiv", id=doi, title=title,
                                text=f"{title}. {abstract}",
                                url=f"https://www.{server}.org/content/{doi}v1",
                                weight=1.0, domain=f"{server}:{cat}"))
            cursor += len(coll)
            total = int((data.get("messages") or [{}])[0].get("total", 0) or 0)
            if cursor >= total:
                break
        log.info("%s %s: %d статей", server, window.week_id,
                 sum(1 for d in docs if d.domain.startswith(server)))
    return docs
