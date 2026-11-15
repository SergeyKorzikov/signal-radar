"""«Занято ли место?» — сколько уже существует игр, модов, репозиториев, видео по теме.

Растущее внимание при пустой полке — это и есть возможность. Проверяем только
отобранные сигналы, чтобы не упираться в лимиты чужих API.
"""
from __future__ import annotations

import logging
import os
import re

from .http import get

log = logging.getLogger("radar.occupancy")


def _roblox(q: str) -> dict | None:
    r = get("https://apis.roblox.com/search-api/omni-search", host_delay=1.0, retries=2,
            params={"searchQuery": q, "sessionId": "1", "pageType": "all"})
    if r is None:
        return None
    games = [g for grp in r.json().get("searchResults", []) if grp.get("contentGroupType") == "Game"
             for g in grp.get("contents", [])]
    words = [w for w in re.findall(r"\w+", q.lower()) if len(w) > 2]
    hits = [g for g in games if all(w in (g.get("name") or "").lower() for w in words)]
    return {"count": len(hits), "top_players": max((g.get("playerCount", 0) for g in hits), default=0),
            "players": sum(g.get("playerCount", 0) for g in hits)}


def _modrinth(q: str) -> dict | None:
    r = get("https://api.modrinth.com/v2/search", params={"query": q, "limit": 1},
            host_delay=0.5, retries=2)
    return {"count": r.json().get("total_hits", 0)} if r is not None else None


def _steam(q: str) -> dict | None:
    r = get("https://store.steampowered.com/api/storesearch/", host_delay=1.0, retries=2,
            params={"term": q, "cc": "us", "l": "en"})
    return {"count": r.json().get("total", 0)} if r is not None else None


def _github(q: str) -> dict | None:
    h = {"Accept": "application/vnd.github+json"}
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    r = get("https://api.github.com/search/repositories", params={"q": f'"{q}"', "per_page": 1},
            headers=h, host_delay=2.5, retries=2)
    return {"count": r.json().get("total_count", 0)} if r is not None else None


def _youtube(q: str) -> dict | None:
    key = os.environ.get("YOUTUBE_API_KEY")
    if not key:
        return None
    r = get("https://www.googleapis.com/youtube/v3/search", host_delay=0.3, retries=2,
            params={"part": "id", "q": q, "type": "video", "maxResults": 1, "key": key})
    return {"count": r.json().get("pageInfo", {}).get("totalResults", 0)} if r is not None else None


def _domain(q: str) -> dict | None:
    name = re.sub(r"[^a-z0-9]", "", q.lower())
    if not 4 <= len(name) <= 24:
        return None
    r = get("https://dns.google/resolve", params={"name": f"{name}.com", "type": "NS"},
            host_delay=0.2, retries=2)
    if r is None:
        return None
    return {"domain": f"{name}.com", "free": r.json().get("Status") == 3}  # 3 = NXDOMAIN


CHECKS = {"roblox": _roblox, "modrinth": _modrinth, "steam": _steam, "github": _github,
          "youtube": _youtube, "domain": _domain}


def check(query: str) -> dict:
    out = {}
    for name, fn in CHECKS.items():
        try:
            res = fn(query)
        except Exception as e:  # noqa: BLE001
            log.warning("%s: %s", name, e)
            res = None
        if res is not None:
            out[name] = res
    return out


def describe(o: dict) -> str:
    """Короткая строка по-русски — для карточки и для нейросети."""
    parts = []
    if "roblox" in o:
        r = o["roblox"]
        parts.append(f"Roblox: {r['count']} игр" + (f", лучший онлайн {r['top_players']}" if r["count"] else ""))
    if "modrinth" in o:
        parts.append(f"моды Minecraft: {o['modrinth']['count']}")
    if "steam" in o:
        parts.append(f"Steam: {o['steam']['count']} игр")
    if "github" in o:
        parts.append(f"GitHub: {o['github']['count']} репозиториев")
    if "youtube" in o:
        parts.append(f"YouTube: {o['youtube']['count']} видео")
    if "domain" in o:
        parts.append(f"домен {o['domain']['domain']} {'свободен' if o['domain']['free'] else 'занят'}")
    return " · ".join(parts)
