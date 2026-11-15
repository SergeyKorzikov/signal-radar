"""Запасной вариант аналитика через Claude API (backend: claude в config.yaml)
и шаблонные разборы, если нейросеть недоступна. По умолчанию работает
своя локальная модель — см. local_llm.py.
"""
from __future__ import annotations

import json
import logging
import os
import re

from .models import SOURCE_LABEL

log = logging.getLogger("radar.analyst")

SYSTEM = """Ты — аналитик «радара слабых сигналов». Твоя задача — среди кандидатов,
найденных статистикой, отобрать настоящие ранние технологические/научные сдвиги —
то, что сейчас на стадии ИИ в 2020–2021 году: строители уже одержимы, мейнстрим ещё
не заметил. Будь строгим и думай самостоятельно, не повторяй хайп.

Отклоняй (keep=false):
- общие фразы и академический шум («proposed framework», «recent advances»);
- имена отдельных продуктов/моделей, если за ними не стоит новый класс технологий;
- то, что уже давно мейнстрим, даже если всплеск;
- спам, NSFW, накрутки, «awesome-списки».
Если два кандидата — одно явление, оставь лучший, другой пометь merge_into.

Для оставленных пиши по-русски, конкретно, без воды. Оценки earn и learn (0–10)
ставь честно: earn — можно ли в ближайший год превратить в продукт/услугу/доход
для одного человека с ИИ-инструментами; learn — насколько выгодно стать
в этом экспертом заранее (на горизонте 2–3 лет). «Ставка на выходные» — что-то,
что реально сделать за 1–2 дня и что проверит гипотезу. «Критерий смерти» —
наблюдаемый признак, по которому сигнал стоит бросить."""

SCHEMA_HINT = """Ответь ТОЛЬКО JSON-объектом вида:
{"signals": [{"term": "<как во входных данных>", "keep": true, "merge_into": null,
  "reject_reason": null, "title_ru": "короткое название по-русски",
  "what": "что это, 1–2 предложения", "why_now": "почему именно сейчас, со ссылкой на цифры",
  "skeptics_vs_fans": "кто скептики и кто фанаты, если видно; иначе null",
  "earn": 0-10, "earn_why": "одно предложение", "learn": 0-10, "learn_why": "одно предложение",
  "cheap_bet": "ставка на выходные", "kill_criteria": "критерий смерти",
  "edge_for_user": "есть ли у пользователя преимущество с его профилем, одно предложение"}],
 "week_summary": "2–3 предложения: общая картина недели"}"""


def _payload(cands: list[dict]) -> list[dict]:
    out = []
    for c in cands:
        out.append({
            "term": c["term"], "label": c["label"], "aliases": c.get("aliases", [])[:4],
            "mentions_this_week": c["now"], "expected_from_baseline": c["expected"],
            "growth_log2": c["growth"], "weekly_rate_per_10k_docs": c["series"],
            "by_source": {SOURCE_LABEL[s]: n for s, n in c["by_source"].items()},
            "github_stars_per_day": c["stars_per_day"],
            "news_articles_4w": c.get("media"), "hn_comments_30d": (c.get("toy") or {}).get("comments"),
            "hn_skeptic_vs_enthusiast": [(c.get("toy") or {}).get("skeptic"),
                                         (c.get("toy") or {}).get("enthusiast")],
            "hn_quotes": [e["text"][:220] for e in (c.get("toy") or {}).get("examples", [])][:2],
            "examples": [f"[{SOURCE_LABEL[e['source']]}] {e['title']}" for e in c["examples"][:8]],
            "score": c["score"], "fit_with_user": c.get("fit"),
        })
    return out


def _parse(text: str) -> dict:
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0) if m else text)


def review(cands: list[dict], profile: str, cfg: dict) -> dict | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key or not cands:
        return None
    try:
        import anthropic
    except ImportError:
        log.warning("пакет anthropic не установлен — разборы будут шаблонными")
        return None
    client = anthropic.Anthropic(api_key=key)
    user = (f"Профиль пользователя:\n{profile}\n\nКандидаты этой недели (уже отсортированы "
            f"статистикой):\n{json.dumps(_payload(cands), ensure_ascii=False, indent=1)}\n\n"
            f"{SCHEMA_HINT}")
    try:
        msg = client.messages.create(model=cfg.get("model", "claude-sonnet-5-5"),
                                     max_tokens=cfg.get("max_tokens", 16000),
                                     system=SYSTEM, messages=[{"role": "user", "content": user}])
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return _parse(text)
    except Exception as e:  # noqa: BLE001 — радар не должен падать из-за аналитика
        log.warning("ИИ-аналитик не сработал: %s", e)
        return None


def apply_review(cands: list[dict], review_: dict | None) -> str | None:
    """Переносит решения аналитика в кандидатов; возвращает сводку недели."""
    if not review_:
        for c in cands:
            c.update(template_writeup(c))
        return None
    by_term = {s.get("term"): s for s in review_.get("signals", [])}
    for c in cands:
        r = by_term.get(c["term"])
        if r is None:
            c.update(template_writeup(c))
            continue
        if not r.get("keep", True) or r.get("merge_into"):
            c["keep"] = False
            c["reject_reason"] = r.get("reject_reason") or (f"дубль «{r['merge_into']}»"
                                                            if r.get("merge_into") else "")
            continue
        c["analysis"] = {k: r.get(k) for k in ("title_ru", "what", "why_now", "skeptics_vs_fans",
                                                "earn_why", "learn_why", "cheap_bet",
                                                "kill_criteria", "edge_for_user")}
        c["analysis"]["by"] = "ai"
        for k in ("earn", "learn"):
            if isinstance(r.get(k), (int, float)):
                c[k] = float(r[k])
    return review_.get("week_summary")


def template_writeup(c: dict) -> dict:
    src = ", ".join(f"{SOURCE_LABEL[s]} — {n}" for s, n in
                    sorted(c["by_source"].items(), key=lambda kv: -kv[1]))
    times = 2 ** c["growth"]
    if c.get("metric_label"):
        why = f"{c['metric_label']}."
    else:
        why = (f"{c['now']} упоминаний за неделю при ожидаемых ≈{c['expected']:.0f} "
               f"(рост ×{times:.1f}). Источники: {src}.")
    if c.get("media") is not None:
        why += f" В новостях за 4 недели: {c['media']}."
    bk = c.get("by_kind") or {"research": 0, "builder": 1}
    research = bk["research"] >= bk["builder"] and not c.get("metric_label")
    bet = ("Прочитайте 2–3 самые свежие статьи из примеров и напишите короткий обзор — "
           "так вы проверите, есть ли за термином реальная идея."
           if research else
           "Запустите самый популярный репозиторий из примеров и соберите на нём "
           "минимальный прототип под свою задачу.")
    return {"analysis": {"title_ru": c["label"], "what": None, "why_now": why,
                         "skeptics_vs_fans": None, "cheap_bet": bet,
                         "kill_criteria": "Через 4 недели рост остановился, а новых "
                                          "репозиториев и статей не прибавилось.",
                         "earn_why": None, "learn_why": None, "edge_for_user": None,
                         "by": "template"}}
