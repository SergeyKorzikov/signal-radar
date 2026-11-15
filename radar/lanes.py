"""Быстрые дорожки: культура и мемы, игры, рынки.

В отличие от науки (где мы сами считаем рост терминов по тысячам статей), здесь
платформы уже отдают готовые списки «новое и растёт». Радар:
1. собирает объекты (Item) из всех таких списков;
2. сравнивает их со своей историей (чего раньше не было — новое);
3. ищет ТЕМЫ — слова, которые одновременно всплыли на разных платформах
   (мем перепрыгнул из Minecraft-модов в Roblox и в поисковые запросы).
   Прыжок между платформами — самый сильный признак.
"""
from __future__ import annotations

import html
import logging
import math
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from .http import get
from .models import SOURCE_LABEL, Window
from .terms import GENERIC, extract, is_common_word

log = logging.getLogger("radar.lanes")

CULTURE, GAMES, MARKETS = "культура и мемы", "игры", "рынки"
SOURCE_LANE = {"wikipedia": CULTURE, "trends": CULTURE, "kym": CULTURE, "urban": CULTURE,
               "roblox": GAMES, "modrinth": GAMES, "steam": GAMES, "itch": GAMES,
               "polymarket": MARKETS, "coingecko": MARKETS}


# Википедия отдаёт просто «самое читаемое», а не «новое»: без своей истории роста не увидеть.
NEEDS_HISTORY = {"wikipedia"}


@dataclass
class Item:
    source: str
    key: str            # стабильный идентификатор внутри источника
    title: str
    value: float        # главная метрика источника (просмотры, онлайн, скачивания, объём)
    url: str
    unit: str = ""      # подпись метрики для карточки
    text: str = ""      # описание — для нейросети
    meta: dict = field(default_factory=dict)

    @property
    def lane(self) -> str:
        return SOURCE_LANE[self.source]


# ------------------------------------------------------------------ источники
_WIKI_SKIP = re.compile(r"^(Main_Page|Special:|Wikipedia:|Portal:|File:|Help:|Category:|Talk:|"
                        r"User:|Template:|\.xxx|Cookie|Pornhub|XXX|Deaths_in_|List_of_)", re.I)


def fetch_wikipedia(window: Window, top: int = 400) -> list[Item]:
    """Самые читаемые статьи английской Википедии за неделю (есть история — можно догружать)."""
    views: dict[str, int] = defaultdict(int)
    day = window.start
    while day < window.end:
        r = get("https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access/"
                f"{day:%Y/%m/%d}", host_delay=1.5, retries=6)
        if r is not None:
            for a in r.json()["items"][0]["articles"]:
                if not _WIKI_SKIP.match(a["article"]):
                    views[a["article"]] += a["views"]
        day += timedelta(days=1)
    best = sorted(views.items(), key=lambda kv: -kv[1])[:top]
    return [Item("wikipedia", k, k.replace("_", " "), v, f"https://en.wikipedia.org/wiki/{k}",
                 "просмотров Википедии за неделю") for k, v in best]


def wiki_summary(title_key: str) -> str:
    r = get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{title_key}", host_delay=0.2,
            retries=2)
    try:
        return (r.json().get("extract") or "")[:400] if r is not None else ""
    except ValueError:
        return ""


def _rss(url: str) -> list[ET.Element]:
    r = get(url, host_delay=0.5, retries=3)
    if r is None:
        return []
    try:
        return ET.fromstring(r.content).findall(".//item")
    except ET.ParseError:
        return []


def _strip(s: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", s or "")).split())


def fetch_trends(geos=("US", "GB", "IN", "BR", "RU", "TR")) -> list[Item]:
    """Google Trends: что прямо сейчас резко ищут в разных странах."""
    ns = {"ht": "https://trends.google.com/trending/rss"}
    agg: dict[str, Item] = {}
    for geo in geos:
        for it in _rss(f"https://trends.google.com/trending/rss?geo={geo}"):
            q = (it.findtext("title") or "").strip().lower()
            traffic = int(re.sub(r"\D", "", it.findtext("ht:approx_traffic", "0", ns)) or 0)
            news = _strip(it.findtext("ht:news_item/ht:news_item_title", "", ns))
            if not q:
                continue
            if q in agg:
                agg[q].value += traffic
                agg[q].meta["geos"].append(geo)
            else:
                agg[q] = Item("trends", q, q, traffic,
                              f"https://trends.google.com/trends/explore?q={q.replace(' ', '%20')}",
                              "поисковых запросов за сутки (оценка Google)", news,
                              {"geos": [geo]})
    return list(agg.values())


def fetch_kym() -> list[Item]:
    """Know Your Meme: новые подтверждённые мемы."""
    out = []
    n = len(_rss("https://knowyourmeme.com/memes.rss"))
    for i, it in enumerate(_rss("https://knowyourmeme.com/memes.rss")):
        link = it.findtext("link") or ""
        title = it.findtext("title") or ""
        out.append(Item("kym", link.rsplit("/", 1)[-1], title, n - i, link,
                        "новая запись в базе мемов", _strip(it.findtext("description"))[:400]))
    return out


def fetch_urban() -> list[Item]:
    """Urban Dictionary: новые слова сленга."""
    r = get("https://api.urbandictionary.com/v0/words_of_the_day", host_delay=0.5, retries=2)
    if r is None:
        return []
    out = []
    for w in r.json().get("list", []):
        word = w.get("word", "")
        out.append(Item("urban", word.lower(), word, w.get("thumbs_up", 0), w.get("permalink", ""),
                        "голосов за слово дня", _strip(w.get("definition", "")).replace("[", "")
                        .replace("]", "")[:300]))
    return out


def fetch_roblox() -> list[Item]:
    """Roblox: игры из подборок «в тренде» и «восходящие» с текущим онлайном."""
    r = get("https://apis.roblox.com/explore-api/v1/get-sorts", params={"sessionId": "1"},
            host_delay=0.5)
    if r is None:
        return []
    seen: dict[str, Item] = {}
    for sort in r.json().get("sorts", []):
        if sort.get("sortId") not in ("top-trending", "up-and-coming"):
            continue
        for g in sort.get("games") or []:
            key = str(g.get("universeId"))
            if key in seen or g.get("isSponsored"):
                continue
            up, down = g.get("totalUpVotes", 0), g.get("totalDownVotes", 0)
            seen[key] = Item("roblox", key, g.get("name", ""), g.get("playerCount", 0),
                             f"https://www.roblox.com/games/{g.get('rootPlaceId')}",
                             "игроков онлайн", g.get("genreL1") or "",
                             {"sort": sort.get("sortId"), "likes": up, "dislikes": down})
    return list(seen.values())


def fetch_modrinth(window: Window, limit: int = 100) -> list[Item]:
    """Modrinth: новые моды Minecraft, набравшие больше всего скачиваний (за 3 недели)."""
    import json as _json
    since = int((window.end - timedelta(days=21)).timestamp())
    r = get("https://api.modrinth.com/v2/search", host_delay=0.5,
            params={"index": "downloads", "limit": limit,
                    "facets": _json.dumps([[f"created_timestamp>{since}"]])})
    if r is None:
        return []
    return [Item("modrinth", h["project_id"], h["title"], h.get("downloads", 0),
                 f"https://modrinth.com/{h.get('project_type', 'mod')}/{h.get('slug')}",
                 "скачиваний мода", (h.get("description") or "")[:300],
                 {"created": h.get("date_created", "")[:10]}) for h in r.json().get("hits", [])]


def fetch_steam(count: int = 100) -> list[Item]:
    """Steam: «популярные новинки»."""
    r = get("https://store.steampowered.com/search/results/", host_delay=1.0,
            params={"filter": "popularnew", "sort_by": "Released_DESC", "json": 1, "count": count})
    if r is None:
        return []
    items = r.json().get("items", [])
    out = []
    for i, g in enumerate(items):
        m = re.search(r"/apps/(\d+)/", g.get("logo", ""))
        app = m.group(1) if m else g.get("name", "")
        out.append(Item("steam", app, g.get("name", ""), len(items) - i,
                        f"https://store.steampowered.com/app/{app}", "место в популярных новинках"))
    return out


def fetch_itch() -> list[Item]:
    """itch.io: новые и популярные инди-игры — здесь жанры появляются раньше Steam."""
    items = _rss("https://itch.io/games/new-and-popular.xml")
    out = []
    for i, it in enumerate(items):
        link = it.findtext("link") or ""
        out.append(Item("itch", link, it.findtext("plainTitle") or it.findtext("title") or "",
                        len(items) - i, link, "место в «новых и популярных»",
                        _strip(it.findtext("description"))[:300]))
    return out


def fetch_polymarket(limit: int = 100) -> list[Item]:
    """Polymarket: рынки предсказаний с наибольшим оборотом (без однодневных спортивных матчей)."""
    r = get("https://gamma-api.polymarket.com/markets", host_delay=0.5,
            params={"limit": limit, "active": "true", "closed": "false",
                    "order": "volume24hr", "ascending": "false"})
    if r is None:
        return []
    now, out = datetime.now(timezone.utc), []
    for m in r.json():
        try:
            end = datetime.fromisoformat((m.get("endDate") or "").replace("Z", "+00:00"))
        except ValueError:
            continue
        if end - now < timedelta(days=5) or re.search(r"\bvs\.?\b", m.get("question", ""), re.I):
            continue  # матч или событие на днях — не тренд
        ch = m.get("oneWeekPriceChange")
        out.append(Item("polymarket", str(m.get("id")), m.get("question", ""),
                        float(m.get("volume24hr") or 0),
                        f"https://polymarket.com/market/{m.get('slug')}",
                        "долларов оборота за сутки", (m.get("description") or "")[:300],
                        {"prices": m.get("outcomePrices"), "week_change": ch,
                         "created": (m.get("createdAt") or "")[:10]}))
    return out


def fetch_coingecko() -> list[Item]:
    """CoinGecko: что сейчас чаще всего ищут среди криптоактивов."""
    r = get("https://api.coingecko.com/api/v3/search/trending", host_delay=2.0, retries=2)
    if r is None:
        return []
    coins = r.json().get("coins", [])
    out = []
    for i, c in enumerate(coins):
        it = c.get("item", {})
        d = it.get("data", {})
        ch = (d.get("price_change_percentage_24h") or {}).get("usd")
        out.append(Item("coingecko", it.get("id", ""), f"{it.get('name')} ({it.get('symbol')})",
                        len(coins) - i, f"https://www.coingecko.com/en/coins/{it.get('id')}",
                        "место в поисковых трендах CoinGecko", "",
                        {"cap_rank": it.get("market_cap_rank"), "change_24h": ch}))
    return out


def collect(window: Window, cfg: dict, live: bool = True) -> list[Item]:
    """live=False — историческая неделя: доступна только Википедия."""
    s = cfg.get("lanes", {}).get("sources", {})
    on = lambda name: s.get(name, True)  # noqa: E731
    items: list[Item] = []
    if on("wikipedia"):
        items += fetch_wikipedia(window)
    if live:
        for name, fn in (("trends", fetch_trends), ("kym", fetch_kym), ("urban", fetch_urban),
                         ("roblox", fetch_roblox), ("steam", fetch_steam), ("itch", fetch_itch),
                         ("polymarket", fetch_polymarket), ("coingecko", fetch_coingecko)):
            if on(name):
                try:
                    items += fn()
                except Exception as e:  # noqa: BLE001 — один источник не должен ронять радар
                    log.warning("%s недоступен: %s", name, e)
        if on("modrinth"):
            items += fetch_modrinth(window)
    by = defaultdict(int)
    for i in items:
        by[i.source] += 1
    log.info("быстрые дорожки: %s", dict(by))
    return items


def _term_index(items: list[Item]) -> tuple[dict[str, list[Item]], dict[str, str]]:
    index: dict[str, list[Item]] = defaultdict(list)
    surface: dict[str, str] = {}
    for it in items:
        for key, surf in extract(it.title).items():
            index[key].append(it)
            surface.setdefault(key, surf)
    return index, surface


def to_history(items: list[Item]) -> dict:
    out: dict[str, dict] = defaultdict(dict)
    for i in items:
        out[i.source][i.key] = round(i.value, 1)
    # сколько объектов несли каждое слово: чтобы вечные жанры («tycoon») не считались новыми темами
    live = [i for i in items if i.source not in NEEDS_HISTORY]
    if live:
        out["_themes"] = {t: len(v) for t, v in _term_index(live)[0].items() if len(v) >= 2}
    return dict(out)


# ------------------------------------------------------------------ оценка
def _pct_rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    for rank, i in enumerate(order):
        out[i] = rank / max(len(values) - 1, 1)
    return out


def _fmt(v: float) -> str:
    if v >= 1e6:
        return f"{v / 1e6:.1f} млн".replace(".", ",")
    if v >= 1e4:
        return f"{v / 1e3:.0f} тыс."
    return f"{v:.0f}"


def _example(i: Item) -> dict:
    return {"source": i.source, "title": i.title[:200], "url": i.url, "snippet": i.text[:320],
            "meta": {"value": f"{_fmt(i.value)} {i.unit}"}}


def build_candidates(items: list[Item], history: list[dict], cfg: dict, tracker: dict,
                     interests: dict) -> list[dict]:
    """history — список недель; у каждой может быть ключ "entities": {source: {key: value}}."""
    from .scoring import fit_score

    lc = cfg.get("lanes", {})
    muted = {t for t, v in tracker.items() if (v or {}).get("status") in ("мёртв", "dead", "mute")}
    past = [w.get("entities") or {} for w in history]
    by_source: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        by_source[it.source].append(it)

    # --- одиночные объекты
    singles: list[dict] = []
    for src, lst in by_source.items():
        weeks_with = [p[src] for p in past if p.get(src)]
        if src in NEEDS_HISTORY and not weeks_with:
            continue
        floor = min(i.value for i in lst) or 1.0
        ranks = _pct_rank([i.value for i in lst])
        for it, rk in zip(lst, ranks):
            prev = [w.get(it.key) for w in weeks_with]
            seen = [v for v in prev if v is not None]
            if weeks_with:
                expected = sum(v if v is not None else floor * 0.5 for v in prev) / len(prev)
                growth = math.log2((it.value + floor) / (expected + floor))
                novelty = 1.0 if not seen else max(0.0, 1 - len(seen) / 4)
                g_part = max(0.0, min(1.0, growth / 3))
            else:  # у источника ещё нет истории: верим платформе, что это «новое и растёт»
                expected, growth, novelty, g_part = 0.0, 0.0, 0.6, 0.5
            if weeks_with and growth < lc.get("min_growth", 0.8):
                continue  # давно в топе и не растёт — не сигнал
            term = f"{src}:{it.key}".lower()
            if term in muted:
                continue
            score = 100 * (0.40 * g_part + 0.30 * rk + 0.30 * novelty) * 0.85
            if src == "wikipedia":
                score *= 0.8  # в топе Википедии в основном новости, а не тренды
            series = [w.get(it.key) or 0 for w in weeks_with] + [it.value]
            singles.append({
                "term": term, "label": it.title, "lane": it.lane, "kind": "объект",
                "now": 1, "value": it.value, "expected": round(expected, 1),
                "growth": round(growth, 2), "novelty": round(novelty, 2),
                "metric_label": f"{_fmt(it.value)} {it.unit}"
                                + (f" · рост ×{2 ** growth:.1f}".replace(".", ",")
                                   if weeks_with and seen else
                                   " · новое в списке" if weeks_with else ""),
                "by_source": {src: 1}, "score": round(score, 1), "spark": series,
                "parts": {"growth": round(g_part, 2), "rank": round(rk, 2), "novelty": round(novelty, 2)},
                "examples": [_example(it)], "_items": [it], "aliases": [],
            })

    # --- темы: одно слово/фраза всплыло на нескольких платформах или во многих объектах
    index, surface = _term_index(items)
    theme_weeks = [p["_themes"] for p in past if p.get("_themes")]
    themes: list[dict] = []
    for term, its in index.items():
        srcs = {i.source for i in its}
        uniq = {(i.source, i.key): i for i in its}
        if len(srcs) < 2 and len(uniq) < lc.get("theme_min_items", 3):
            continue
        if srcs == {"wikipedia"}:
            continue  # общие слова в названиях статей («TV series») — не тема
        words = term.split()
        if all(w in GENERIC or is_common_word(w, 3.6) for w in words):
            continue  # «new», «simulator», «game» и т.п.
        if f"тема:{term}" in muted:
            continue
        usual = sum(w.get(term, 0) for w in theme_weeks) / len(theme_weeks) if theme_weeks else 0
        if theme_weeks and len(uniq) < 1.5 * usual + 1:
            continue  # это слово всегда в списках (жанр, вечная тема) — не новое
        lanes = defaultdict(float)
        for i in uniq.values():
            lanes[i.lane] += 1
        lane = max(lanes, key=lanes.get)
        hops = len(srcs)
        score = min(100.0, 38 + 17 * (hops - 1) + 4 * min(len(uniq), 6)
                    + (6 if len(words) > 1 else 0))
        top = sorted(uniq.values(), key=lambda i: -i.value)
        themes.append({
            "term": f"тема:{term}", "label": surface[term], "lane": lane, "kind": "тема",
            "now": len(uniq), "expected": 0.0, "growth": 0.0, "novelty": 0.7, "hops": hops,
            "metric_label": f"{len(uniq)} объектов на {hops} "
                            f"{'платформе' if hops == 1 else 'платформах'}: "
                            + ", ".join(SOURCE_LABEL[s] for s in sorted(srcs)),
            "by_source": {s: sum(1 for i in uniq.values() if i.source == s) for s in srcs},
            "score": round(score, 1),
            "parts": {"hops": round(min(1, (hops - 1) / 3), 2), "items": round(min(1, len(uniq) / 6), 2)},
            "examples": [_example(i) for i in top[:8]], "_items": top, "aliases": [],
        })

    # тема поглощает вложенные темы с теми же объектами и одиночные объекты
    themes.sort(key=lambda c: (-c["score"], -len(c["term"])))
    kept: list[dict] = []
    for t in themes:
        ids = {(i.source, i.key) for i in t["_items"]}
        dup = None
        for k in kept:
            kid = {(i.source, i.key) for i in k["_items"]}
            if len(ids & kid) / min(len(ids), len(kid)) >= 0.8:
                dup = k
                break
        if dup:
            dup["aliases"].append(t["label"])
        else:
            kept.append(t)
    covered = {(i.source, i.key) for t in kept for i in t["_items"]}
    singles = [s for s in singles if (s["_items"][0].source, s["_items"][0].key) not in covered]

    out = kept + singles
    for c in out:
        titles = [i.title + " " + i.text for i in c.pop("_items")][:20]
        c["fit"], c["fit_domain"] = fit_score(c["label"], titles, interests)
        fb = (tracker.get(c["term"]) or {}).get("rating")
        if fb in ("+", "+1", 1, "👍"):
            c["fit"] = min(1.0, c["fit"] + 0.3)
        c["earn"] = {GAMES: 6.0, CULTURE: 5.0, MARKETS: 4.0}[c["lane"]]
        c["learn"] = 3.0
        c["stars_per_day"] = 0
    return out


def pick_pool(cands: list[dict], per_lane: int) -> list[dict]:
    """Лучшие кандидаты каждой дорожки — на проверку нейросетью."""
    pool = []
    for lane in (CULTURE, GAMES, MARKETS):
        lane_c = sorted((c for c in cands if c["lane"] == lane), key=lambda c: -c["score"])
        taken, per_src = 0, defaultdict(int)
        for c in lane_c:  # не больше трёх одиночных объектов с одной платформы
            src = next(iter(c["by_source"])) if c["kind"] == "объект" else None
            if src and per_src[src] >= 3:
                continue
            if src:
                per_src[src] += 1
            pool.append(c)
            taken += 1
            if taken >= per_lane:
                break
    return pool
