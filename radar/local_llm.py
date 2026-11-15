"""Своя небольшая нейросеть внутри GitHub Actions — без внешних API и ключей.

Модель (по умолчанию Qwen3-4B-Instruct, ~2,5 ГБ) скачивается с Hugging Face
один раз и дальше берётся из кэша. Работает на обычном процессоре через llama.cpp.

Две задачи:
1. Фильтр: для каждого кандидата решить, технология это или шум.
2. Разбор: для отобранных сигналов написать по-русски, что это и что с этим делать.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from .http import _session
from .models import SOURCE_LABEL

log = logging.getLogger("radar.llm")
CACHE = Path(os.environ.get("RADAR_MODEL_DIR", Path.home() / ".cache" / "radar-models"))

KINDS_KEEP = ("технология, метод или инструмент", "новые данные или научный проект",
              "мем, персонаж или интернет-явление", "игра, мод или игровой жанр",
              "рыночная возможность")
KINDS = KINDS_KEEP + ("устоявшаяся область", "шаблонная фраза или обычное слово",
                      "имя, компания или разовая новость")
CHANNELS = ["игра в Roblox", "короткие видео", "сайт или веб-инструмент", "бот или расширение",
            "мерч или цифровые товары", "мод или ассет", "рыночная ставка",
            "экспертиза: обзор, курс, консалтинг", "продукт на основе технологии"]
STAGES = ["зарождение", "разгон", "пик", "спад"]
WINDOWS = ["дни", "недели", "месяцы", "годы"]

# Порядок полей важен: модель сначала объясняет, потом решает (так маленькие модели ошибаются реже).
FILTER_SCHEMA = {
    "type": "object",
    "properties": {
        "what": {"type": "string"},
        "kind": {"type": "string", "enum": list(KINDS)},
        "age": {"type": "string", "enum": ["новое (последние 1–3 года)", "старое (известно давно)"]},
        "title_ru": {"type": "string"},
    },
    "required": ["what", "kind", "age", "title_ru"],
}

WRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "what": {"type": "string"},
        "why_now": {"type": "string"},
        "stage": {"type": "string", "enum": STAGES},
        "window": {"type": "string", "enum": WINDOWS},
        "applications": {
            "type": "array", "minItems": 2, "maxItems": 3,
            "items": {"type": "object",
                      "properties": {"channel": {"type": "string", "enum": CHANNELS},
                                     "idea": {"type": "string"},
                                     "first_step": {"type": "string"}},
                      "required": ["channel", "idea", "first_step"]}},
        "kill_criteria": {"type": "string"},
        "earn_why": {"type": "string"},
        "earn": {"type": "integer", "minimum": 0, "maximum": 10},
        "learn_why": {"type": "string"},
        "learn": {"type": "integer", "minimum": 0, "maximum": 10},
    },
    "required": ["what", "why_now", "stage", "window", "applications", "kill_criteria",
                 "earn_why", "earn", "learn_why", "learn"],
}

FILTER_SYSTEM = """Ты — строгий фильтр радара ранних трендов. Тебе дают объект или термин, который
начал расти в одной из дорожек (наука, технологии, культура и мемы, игры, рынки), и примеры.
Заполни поля по порядку:
what — что это такое, 5–12 слов по-русски, только по примерам;
kind — тип:
  «технология, метод или инструмент» — конкретная вещь, которую можно применить или построить;
  «новые данные или научный проект» — новый набор данных, телескоп, атлас, крупный проект;
  «мем, персонаж или интернет-явление» — мем, вирусный персонаж, формат, сленг, челлендж, фэндом;
  «игра, мод или игровой жанр» — новая игра, мод, механика или жанр, набирающие игроков;
  «рыночная возможность» — рынок предсказаний или актив с необычным ростом интереса;
  «устоявшаяся область» — большая давно существующая тема;
  «шаблонная фраза или обычное слово» — фраза или слово без конкретного явления за ним;
  «имя, компания или разовая новость» — человек, спортивный матч, выборы, смерть знаменитости,
    премьера фильма или сериала, альбом, инцидент, ставки на исход матча, выборов или на курс
    к дате: то, о чём пишут новости, но на чём нельзя ничего построить;
age — новое это (появилось или стало популярным за последние 1–3 года) или известно давно;
title_ru — короткое понятное название по-русски, 2–5 слов."""

FILTER_EXAMPLES = [
    ("neural radiance field", "наука", "Neural Radiance Fields for view synthesis | NeRF in the wild",
     {"what": "метод 3D-реконструкции сцены нейросетью по фотографиям",
      "kind": "технология, метод или инструмент", "age": "новое (последние 1–3 года)",
      "title_ru": "Нейронные поля излучения (NeRF)"}),
    ("experimental results", "наука", "Experimental results on graph learning | Robust control: experimental results",
     {"what": "раздел статьи с результатами экспериментов",
      "kind": "шаблонная фраза или обычное слово", "age": "старое (известно давно)",
      "title_ru": "Экспериментальные результаты"}),
    ("skibidi", "культура и мемы", "[Roblox] Skibidi Tower Defense | [Know Your Meme] Skibidi Toilet | [Google Trends] skibidi meaning",
     {"what": "вирусный сериал-мем про поющие головы, разошедшийся по играм",
      "kind": "мем, персонаж или интернет-явление", "age": "новое (последние 1–3 года)",
      "title_ru": "Мем Skibidi Toilet"}),
    ("Will Team A win the 2027 league title?", "рынки", "[Polymarket] Will Team A win the 2027 league title? (400 тыс. долларов оборота за сутки)",
     {"what": "ставки на исход спортивного турнира",
      "kind": "имя, компания или разовая новость", "age": "старое (известно давно)",
      "title_ru": "Ставки на чемпиона лиги"}),
    ("Champions League final", "культура и мемы", "[Википедия] 2026 UEFA Champions League final | [Google Trends] champions league final",
     {"what": "футбольный матч, финал турнира этого года",
      "kind": "имя, компания или разовая новость", "age": "старое (известно давно)",
      "title_ru": "Финал Лиги чемпионов"}),
]

WRITE_SYSTEM = """Ты — аналитик радара ранних трендов. Пиши по-русски, конкретно, коротко.
ГЛАВНОЕ ПРАВИЛО: опирайся только на данные и примеры, которые тебе дали. Не выдумывай даты,
организации, цифры и названия, которых нет в данных. Если чего-то не видно — так и напиши.
Не пиши про рост в процентах или разах, если этого числа нет в поле «метрика».
Примеры с разных платформ могут оказаться разными вещами с одинаковым словом в названии
(мем и одноимённый фильм): описывай то, о чём большинство примеров, а лишнее не смешивай.
Поля (заполняй по порядку):
what — что это, 1–2 предложения простым языком, строго по описаниям из примеров;
why_now — почему всплыло сейчас: используй цифры из данных;
stage — стадия тренда: зарождение (знают единицы), разгон (быстро растёт), пик, спад;
window — сколько осталось до того, как это станет общим местом: дни, недели, месяцы или годы
  (мемы — обычно дни или недели, игры — недели или месяцы, технологии — месяцы или годы);
applications — 2–3 РАЗНЫХ способа применить тренд. Для каждого: channel — канал из списка;
  idea — что именно сделать, одно предложение (должно соответствовать каналу: для канала
  «игра в Roblox» — игра в Roblox, а не мод); first_step — первое действие на 1–2 дня.
  Смотри на поле «занятость»: где уже много игр или проектов — туда поздно, где пусто — туда и идти.
  Выбирай каналы, подходящие типу тренда: для мема — игра, видео, мерч, мод; для технологии —
  продукт, инструмент, экспертиза; для рынка — рыночная ставка или инструмент;
kill_criteria — признак провала: что должно случиться, чтобы бросить сигнал;
earn_why и earn (0–10) — можно ли одному человеку с ИИ-инструментами быстро получить с этого
  доход. Фундаментальная наука без прямого применения — 0–3; тренд с пустой полкой — 6–10;
learn_why и learn (0–10) — выгодно ли стать в этом экспертом заранее (2–3 года)."""


def ensure_model(repo: str, filename: str) -> Path:
    """Скачивает GGUF-модель с Hugging Face (один раз)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / filename
    if path.exists() and path.stat().st_size > 100_000_000:
        return path
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    log.info("скачиваю модель %s …", url)
    tmp = path.with_suffix(".part")
    with _session.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 22):
                f.write(chunk)
    tmp.rename(path)
    return path


class LocalAnalyst:
    def __init__(self, cfg: dict):
        from llama_cpp import Llama  # тяжёлый импорт — только когда нужно

        path = ensure_model(cfg["repo"], cfg["file"])
        t = time.time()
        self.llm = Llama(model_path=str(path), n_ctx=cfg.get("n_ctx", 4096),
                         n_threads=cfg.get("threads") or os.cpu_count(), verbose=False)
        log.info("модель загружена за %.0fс", time.time() - t)

    def _ask(self, system: str, user: str, schema: dict, max_tokens: int,
             shots: list[tuple[str, dict]] | None = None) -> dict | None:
        messages = [{"role": "system", "content": system}]
        for u, a in shots or []:
            messages += [{"role": "user", "content": u},
                         {"role": "assistant", "content": json.dumps(a, ensure_ascii=False)}]
        messages.append({"role": "user", "content": user})
        try:
            out = self.llm.create_chat_completion(
                messages=messages, temperature=0.1, max_tokens=max_tokens,
                response_format={"type": "json_object", "schema": schema})
            return json.loads(out["choices"][0]["message"]["content"])
        except Exception as e:  # noqa: BLE001
            log.warning("модель не ответила: %s", e)
            return None

    # ---------------------------------------------------------------- фильтр
    @staticmethod
    def _filter_prompt(term: str, lane: str, titles: str) -> str:
        return f"Дорожка: {lane}\nОбъект: «{term}»\nПримеры: {titles}"

    def filter(self, cands: list[dict]) -> None:
        shots = [(self._filter_prompt(t, lane, ex), a) for t, lane, ex, a in FILTER_EXAMPLES]
        for c in cands:
            titles = " | ".join(f"[{SOURCE_LABEL[e['source']]}] {e['title'][:110]}"
                                for e in c["examples"][:6])
            user = self._filter_prompt(c["label"], c.get("lane", "наука"), titles)
            snippet = next((e["snippet"] for e in c["examples"] if e.get("snippet")), "")
            if snippet:
                user += f"\nОписание: {snippet[:300]}"
            if c.get("metric_label"):
                user += f"\nМетрика: {c['metric_label']}"
            t = time.time()
            r = self._ask(FILTER_SYSTEM, user, FILTER_SCHEMA, 120, shots)
            log.info("фильтр «%s» → %s / %s (%.0fс)", c["label"], r and r.get("kind"),
                     r and r.get("age"), time.time() - t)
            if not r:
                continue
            c["llm_title"] = (r.get("title_ru") or "").strip()
            c["llm_what"] = r.get("what")
            c["llm_kind"] = r.get("kind")
            new = str(r.get("age", "")).startswith("новое")
            if r.get("kind") not in KINDS_KEEP or not new:
                c["keep"] = False
                c["reject_reason"] = f"{r.get('kind')}, {'новое' if new else 'известно давно'}: {r.get('what')}"

    # ---------------------------------------------------------------- разбор
    def write(self, c: dict, profile: str = "") -> dict | None:
        from . import occupancy

        toy = c.get("toy") or {}
        data = {
            "дорожка": c.get("lane"), "тип": c.get("llm_kind"),
            "объект": c["label"], "название": c.get("llm_title") or c["label"],
            "метрика": c.get("metric_label") or (
                f"{c['now']} упоминаний за неделю при ожидаемых {c['expected']}, "
                f"рост в {round(2 ** c['growth'], 1)} раза"),
            "по_источникам": {SOURCE_LABEL[s]: n for s, n in c["by_source"].items()},
            "занятость": occupancy.describe(c.get("occupancy") or {}) or None,
            "звёзд_GitHub_в_сутки": c.get("stars_per_day") or None,
            "HN_скептиков_и_энтузиастов": [toy.get("skeptic"), toy.get("enthusiast")] if toy else None,
            "примеры": [f"[{SOURCE_LABEL[e['source']]}] {e['title'][:150]}"
                        + (f" ({e['meta']['value']})" if (e.get("meta") or {}).get("value") else "")
                        + (f" — {e['snippet'][:260]}" if e.get("snippet") else "")
                        for e in c["examples"][:6]],
        }
        data = {k: v for k, v in data.items() if v is not None}
        # профиль пользователя маленькой модели не даём: она начинает притягивать к нему
        # всё подряд. Связь с интересами считаем в коде (поле fit).
        user = f"Данные сигнала:\n{json.dumps(data, ensure_ascii=False, indent=1)}"
        t = time.time()
        r = self._ask(WRITE_SYSTEM, user, WRITE_SCHEMA, 900)
        log.info("разбор «%s» (%.0fс)", c["label"], time.time() - t)
        return r

    def chat(self, messages: list[dict], max_tokens: int = 700) -> str | None:
        try:
            out = self.llm.create_chat_completion(messages=messages, temperature=0.3,
                                                  max_tokens=max_tokens)
            return out["choices"][0]["message"]["content"].strip()
        except Exception as e:  # noqa: BLE001
            log.warning("модель не ответила: %s", e)
            return None

    def summarize(self, chosen: list[dict]) -> str | None:
        lines = "\n".join(f"- {c['analysis'].get('title_ru') or c['label']}: "
                          f"{c['analysis'].get('what') or ''}" for c in chosen)
        r = self._ask("Ты — аналитик радара ранних сигналов. Пиши по-русски, коротко.",
                      f"Сигналы этой недели:\n{lines}\n\nОпиши общую картину недели в 2–3 "
                      f"предложениях: что объединяет сигналы и на что смотреть в первую очередь.",
                      {"type": "object", "properties": {"summary": {"type": "string"}},
                       "required": ["summary"]}, 300)
        return (r or {}).get("summary")


DOMAIN_RU = {"neuroscience": "нейронаук", "ai": "ИИ", "learning": "обучения и чтения",
             "games": "игр", "trading": "трейдинга"}


def edge_text(c: dict) -> str | None:
    fit, dom = c.get("fit", 0), DOMAIN_RU.get(c.get("fit_domain", ""), c.get("fit_domain"))
    if fit >= 0.6:
        return f"Прямо в вашей области: {dom}. Здесь у вас фора перед теми, кто заходит с нуля."
    if fit >= 0.15:
        return f"Пересекается с вашей областью: {dom}. Стоит смотреть, как это переносится к вам."
    return None


def apply_write(c: dict, r: dict | None) -> None:
    from .analyst import template_writeup

    if not r:
        c.update(template_writeup(c))
        if c.get("llm_title"):
            c["analysis"]["title_ru"] = c["llm_title"]
        return
    apps = [a for a in (r.get("applications") or []) if isinstance(a, dict) and a.get("idea")]
    c["analysis"] = {k: r.get(k) for k in ("what", "why_now", "earn_why", "learn_why",
                                            "kill_criteria", "stage", "window")}
    c["analysis"]["applications"] = apps
    c["analysis"]["cheap_bet"] = apps[0].get("first_step") if apps else None
    c["analysis"].update({"title_ru": c.get("llm_title") or c["label"],
                          "skeptics_vs_fans": None, "by": "local"})
    for k in ("earn", "learn"):
        if isinstance(r.get(k), (int, float)):
            c[k] = float(max(0, min(10, r[k])))
