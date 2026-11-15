"""Оценка «ранности» сигналов.

Идея: ищем термины, у которых ускоряется активность там, где люди *делают*
(статьи, код, модели), при почти нулевом шуме там, где о них *говорят* (СМИ).
"""
from __future__ import annotations

import math
import re
from collections import Counter

from .models import BUILDER, RESEARCH, SOURCE_KIND, TALK
from .terms import is_common_word

KINDS = (RESEARCH, BUILDER, TALK)
COMMERCIAL = re.compile(r"\b(startup|pricing|saas|customer|launch|api|paid|revenue|"
                        r"product|business|enterprise|platform|service)\b", re.I)


# ------------------------------------------------------------------ helpers
def _slope(ys: list[float]) -> float:
    n = len(ys)
    if n < 2:
        return 0.0
    mx = (n - 1) / 2
    my = sum(ys) / n
    num = sum((i - mx) * (y - my) for i, y in enumerate(ys))
    den = sum((i - mx) ** 2 for i in range(n))
    return num / den if den else 0.0


def _percentiles(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    n = max(len(values) - 1, 1)
    for rank, i in enumerate(order):
        out[i] = rank / n
    return out


def fit_score(term: str, titles: list[str], interests: dict[str, list[str]]) -> tuple[float, str]:
    """Насколько сигнал из ваших областей: 1 — прямо в термине, меньше — по примерам."""
    best, best_dom = 0.0, ""
    t = term.lower()
    for dom, kws in interests.items():
        if any(k in t for k in kws):
            return 1.0, dom
        hits = sum(1 for title in titles if any(k in title.lower() for k in kws))
        share = hits / max(len(titles), 1)
        if share > best:
            best, best_dom = share, dom
    return round(min(best, 0.9), 2), best_dom


# ------------------------------------------------------------------ stage 1
def build_candidates(cur, history: list[dict], cfg: dict, tracker: dict) -> tuple[list[dict], list[dict]]:
    """Считает метрики по истории и возвращает (кандидаты, «уже волна»)."""
    sc = cfg["scoring"]
    base = history[-sc["baseline_weeks"]:]
    # источники с базовой линией: есть в текущей неделе и хотя бы в половине базовых
    covered = [s for s in cur.n_docs
               if sum(1 for w in base if w["n_docs"].get(s)) >= max(1, len(base) // 2)]
    base_n = {s: sum(w["n_docs"].get(s, 0) for w in base) for s in covered}
    first_seen: dict[str, int] = {}
    for i, w in enumerate(history):
        for t in w["terms"]:
            first_seen.setdefault(t, i)
    muted = {t for t, v in tracker.items() if (v or {}).get("status") in ("мёртв", "dead", "mute")}

    # Пока у источника нет базовой линии (например, Hugging Face в первые недели),
    # он не участвует ни в росте, ни в разнообразии — только даёт примеры.
    counted = covered or list(cur.n_docs)
    # Рост считаем только там, где люди ДЕЛАЮТ (статьи, код, модели). Обсуждения на HN
    # дают разнообразие и «тест игрушки», но сами по себе — это новости, а не технология.
    core = [s for s in counted if SOURCE_KIND[s] != TALK] or counted
    n_now = sum(cur.n_docs[s] for s in core) or 1
    base_tot = [sum(w["n_docs"].get(s, 0) for s in core) for w in base]
    min_title = sc.get("min_title_share", 0.15)
    min_spec = sc.get("min_specificity", 0.05)
    min_growth = sc.get("min_growth", 1.0)

    cands, wave = [], []
    min_now = sc["min_docs_now"]
    for term, df in cur.df.items():
        if term in muted:
            continue
        dfc = {s: n for s, n in df.items() if s in counted}
        core_now = sum(n for s, n in dfc.items() if s in core)
        if core_now < min_now:
            continue
        if " " not in term and is_common_word(term):
            continue  # обычное английское слово само по себе — не технология
        focus = cur.title_share(term)
        if focus < min_title:
            continue  # термин только мелькает в тексте, ни одна работа не посвящена ему
        spec = cur.specificity(term)
        if spec < min_spec:
            continue  # случайное сочетание частых слов («single machine»)
        total_now = sum(dfc.values())

        # ряд по неделям: частота на 10 000 документов учитываемых источников.
        # В истории хранятся термины с ≥2 упоминаниями, поэтому отсутствие термина
        # означает «0 или 1» — считаем 0,5, чтобы не принимать редкое за новое.
        weekly = []
        for w, n in zip(base, base_tot):
            row_t = w["terms"].get(term)
            weekly.append((sum(row_t.get(s, 0) for s in core) if row_t else 0.5, n))
        if covered and weekly:
            rate = sum(c for c, _ in weekly) / (sum(n for _, n in weekly) or 1)
            expected = rate * n_now
            growth = math.log2((core_now + 1) / (expected + 1))
        else:  # самый первый запуск без истории
            expected = 0.0
            growth = math.log2(core_now + 1) - 1
        series = [1e4 * (c if c != 0.5 else 0) / (n or 1) for c, n in weekly] + [1e4 * core_now / n_now]
        by_kind_series = {k: [] for k in KINDS}
        for w in base + [{"terms": {term: dict(df)}}]:
            row_t = w["terms"].get(term, {})
            for k in KINDS:
                by_kind_series[k].append(sum(v for s, v in row_t.items() if SOURCE_KIND.get(s) == k))
        logs = [math.log(v + 0.5) for v in series]
        recent = _slope(logs[-4:])
        earlier = _slope(logs[-8:-4]) if len(logs) >= 8 else 0.0
        acceleration = recent - 0.5 * earlier

        kinds = Counter()
        for s, c in dfc.items():
            kinds[SOURCE_KIND[s]] += c
        # тип источника засчитывается, если в нём хотя бы 2 упоминания (одна статья — случайность)
        n_kinds = sum(1 for k in KINDS if kinds[k] >= 2)
        if n_kinds == 0 or (n_kinds == 1 and total_now < sc.get("single_kind_factor", 2.5) * min_now):
            continue
        # одни разговоры без статей и кода — это новости (CEO, сбои, скандалы), а не технология
        if kinds[RESEARCH] < 2 and kinds[BUILDER] < 2:
            continue

        idx = cur.postings[term]
        builder_w = sum(cur.docs[i].weight for i in idx
                        if cur.docs[i].kind == BUILDER and cur.docs[i].source in counted)
        stars_day = sum(cur.docs[i].meta.get("stars_per_day", 0) for i in idx
                        if cur.docs[i].source == "github")
        age = len(history) - first_seen.get(term, len(history))
        novelty = max(0.0, 1.0 - age / max(sc["baseline_weeks"], 1))

        row = {
            "term": term, "label": cur.surface.get(term, term), "now": total_now,
            "by_source": dict(df), "by_kind": {k: kinds[k] for k in KINDS},
            "expected": round(expected, 2), "growth": round(growth, 3),
            "acceleration": round(acceleration, 3), "n_kinds": n_kinds,
            "builder_weight": round(builder_w, 2), "stars_per_day": round(stars_day, 1),
            "novelty": round(novelty, 2), "focus": round(focus, 2), "specificity": round(min(spec, 1), 2),
            "weeks_seen": age + 1, "series": [round(v, 2) for v in series],
            "series_by_kind": by_kind_series,
            "series_weeks": [w.get("week", "") for w in base] + ["now"],
        }
        if expected > sc["mainstream_expected"]:
            if growth > 0.6:
                wave.append(row)
            continue
        if growth < min_growth:
            continue
        cands.append(row)

    wave.sort(key=lambda r: r["growth"], reverse=True)
    return cands, wave


def score(cands: list[dict], cfg: dict) -> None:
    """Итоговый балл 0–100. media_gap и toy_test, если не измерены, — нейтральные."""
    if not cands:
        return
    w = cfg["scoring"]["weights"]
    pg = _percentiles([c["growth"] for c in cands])
    pa = _percentiles([c["acceleration"] for c in cands])
    pb = _percentiles([c["builder_weight"] for c in cands])
    for c, g, a, b in zip(cands, pg, pa, pb):
        diversity = (c["n_kinds"] - 1) / 2
        parts = {
            "growth": g, "acceleration": a, "diversity": diversity,
            "builders": b if c["by_kind"][BUILDER] else 0.0,
            "media_gap": c.get("media_gap", 0.5), "toy_test": c.get("toy_score", 0.2),
            "novelty": c["novelty"], "focus": c.get("focus", 0.5),
        }
        c["parts"] = {k: round(v, 3) for k, v in parts.items()}
        raw = 100 * sum(w[k] * parts[k] for k in w)
        # одиночные слова чаще оказываются шумом («eval», «turbo»), чем фразы
        if " " not in c["term"]:
            raw *= cfg["scoring"].get("unigram_penalty", 0.85)
        c["score"] = round(raw, 1)


def dedupe(cands: list[dict], postings: dict[str, list[int]], overlap: float = 0.8) -> list[dict]:
    """Сливаем фразы об одном и том же («neural dust implant», «dust implant firmware»).

    Две фразы — одно явление, если у них есть общее слово и документы одной почти
    целиком входят в документы другой.
    """
    kept: list[dict] = []
    for c in sorted(cands, key=lambda r: r["score"], reverse=True):
        w1 = set(c["term"].split())
        s1 = set(postings.get(c["term"], []))
        dup_i = None
        for i, k in enumerate(kept):
            s2 = set(postings.get(k["term"], []))
            if not (s1 and s2):
                continue
            share = len(s1 & s2) / min(len(s1), len(s2))
            # общее слово + сильное пересечение, либо почти те же самые документы без общего слова
            # (несколько фраз из одного и того же расплодившегося шаблона репозитория)
            if (w1 & set(k["term"].split()) and share >= overlap) or share >= 0.9:
                dup_i = i
                break
        if dup_i is None:
            kept.append(c)
            continue
        k = kept[dup_i]
        # если новая фраза — уточнение старой и почти так же частотна, оставляем её
        nested = f" {k['term']} " in f" {c['term']} "
        if nested and c["now"] >= 0.6 * k["now"]:
            c["aliases"] = k.get("aliases", []) + [k["label"]] + c.get("aliases", [])
            c["score"] = max(c["score"], k["score"])
            kept[dup_i] = c
        else:
            k.setdefault("aliases", []).append(c["label"])
    return kept


def enrich_static(c: dict, cur, interests: dict, tracker: dict) -> None:
    """Примеры документов, соответствие интересам, деньги/обучение (эвристика)."""
    idx = cur.postings[c["term"]]
    docs = sorted((cur.docs[i] for i in idx), key=lambda d: d.weight, reverse=True)
    picked, per_src = [], Counter()
    for d in docs:
        if per_src[d.source] >= 3:
            continue
        per_src[d.source] += 1
        snippet = d.text[len(d.title):].lstrip(". ")[:320] if len(picked) < 3 else ""
        picked.append({"source": d.source, "title": d.title[:220], "url": d.url, "snippet": snippet,
                       "meta": {k: v for k, v in d.meta.items() if k in ("stars", "stars_per_day",
                                                                          "points", "likes")}})
        if len(picked) >= 9:
            break
    c["examples"] = picked
    titles = [d.title for d in docs[:30]]
    fit, dom = fit_score(c["term"], titles, interests)
    fb = (tracker.get(c["term"]) or {}).get("rating")
    if fb in ("+", "+1", 1, "👍"):
        fit = min(1.0, fit + 0.3)
    c["fit"], c["fit_domain"] = fit, dom
    c["domains"] = [d for d, _ in Counter(d.domain for d in docs if d.domain).most_common(3)]

    total = max(c["now"], 1)
    shares = {k: c["by_kind"][k] / total for k in KINDS}
    commercial = sum(1 for d in docs[:40] if d.kind != RESEARCH and COMMERCIAL.search(d.text))
    earn = 10 * (0.45 * shares[BUILDER] + 0.25 * shares[TALK]
                 + 0.15 * min(1.0, c["stars_per_day"] / 150) + 0.15 * min(1.0, commercial / 5))
    learn = 10 * (0.6 * shares[RESEARCH] + 0.4 * c["novelty"])
    c["earn"], c["learn"] = round(earn, 1), round(learn, 1)


def toy_score(t: dict) -> float:
    s, e = t.get("skeptic", 0), t.get("enthusiast", 0)
    if s and e:
        return round(min(1.0, (2 * s * e / (s + e)) / 2), 2)
    if e:
        return 0.35
    if s:
        return 0.25
    return 0.2


def media_gap(media: int | None, now: int) -> float:
    if media is None:
        return 0.5
    activity = max(now, 1) * 4  # ≈ за 4 недели, как и окно GDELT
    return round(1 / (1 + media / (3 * activity)), 3)


def select(cands: list[dict], cfg: dict, fit_bonus: float) -> tuple[list[dict], list[dict]]:
    sel = cfg["selection"]
    for c in cands:
        c["rank"] = c["score"] * (1 + fit_bonus * c.get("fit", 0))
    pool = [c for c in cands if c.get("keep", True)]
    main = sorted(pool, key=lambda c: c["rank"], reverse=True)[:sel["main"]]
    chosen = {c["term"] for c in main}
    outside = [c for c in sorted(pool, key=lambda c: c["score"], reverse=True)
               if c["term"] not in chosen and c.get("fit", 0) < 0.15][:sel["outside_bubble"]]
    return main, outside
