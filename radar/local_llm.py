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

KINDS_KEEP = ("технология, метод или инструмент", "новые данные или научный проект")
KINDS = KINDS_KEEP + ("устоявшаяся область", "шаблонная фраза или обычное слово",
                      "имя, компания или новость")

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
        "cheap_bet": {"type": "string"},
        "kill_criteria": {"type": "string"},
        "earn_why": {"type": "string"},
        "earn": {"type": "integer", "minimum": 0, "maximum": 10},
        "learn_why": {"type": "string"},
        "learn": {"type": "integer", "minimum": 0, "maximum": 10},
    },
    "required": ["what", "why_now", "cheap_bet", "kill_criteria", "earn_why", "earn",
                 "learn_why", "learn"],
}

FILTER_SYSTEM = """Ты — строгий фильтр радара ранних технологических сигналов. Тебе дают термин,
который стал чаще встречаться в научных статьях и коде, и примеры работ с ним.
Заполни поля по порядку:
what — что это такое, 5–12 слов по-русски, только по примерам;
kind — к какому типу относится термин:
  «технология, метод или инструмент» — конкретная вещь, которую можно применить или построить;
  «новые данные или научный проект» — новый набор данных, телескоп, атлас, крупный проект;
  «устоявшаяся область» — большая давно существующая тема (обработка сигналов, энергосистемы,
    управление движением, синтез белка);
  «шаблонная фраза или обычное слово» — фраза, которой просто пишут статьи;
  «имя, компания или новость»;
age — новое это (появилось или стало популярным за последние 1–3 года) или известно давно;
title_ru — короткое понятное название по-русски, 2–5 слов."""

FILTER_EXAMPLES = [
    ("neural radiance field", "Neural Radiance Fields for view synthesis | NeRF in the wild",
     {"what": "метод 3D-реконструкции сцены нейросетью по фотографиям",
      "kind": "технология, метод или инструмент", "age": "новое (последние 1–3 года)",
      "title_ru": "Нейронные поля излучения (NeRF)"}),
    ("experimental results", "Experimental results on graph learning | Robust control: experimental results",
     {"what": "раздел статьи с результатами экспериментов",
      "kind": "шаблонная фраза или обычное слово", "age": "старое (известно давно)",
      "title_ru": "Экспериментальные результаты"}),
    ("image segmentation", "Image segmentation of MRI | Lightweight image segmentation for drones",
     {"what": "разделение изображения на области по объектам",
      "kind": "устоявшаяся область", "age": "старое (известно давно)",
      "title_ru": "Сегментация изображений"}),
]

WRITE_SYSTEM = """Ты — аналитик радара ранних технологических сигналов. Пиши по-русски, конкретно, коротко.
ГЛАВНОЕ ПРАВИЛО: опирайся только на данные и примеры, которые тебе дали. Не выдумывай даты,
организации, цифры и названия, которых нет в данных. Если чего-то не видно — так и напиши.
Поля (заполняй по порядку):
what — что это, 1–2 предложения простым языком, по аннотациям примеров;
why_now — почему всплыло сейчас: используй цифры из данных (рост, источники);
cheap_bet — одно конкретное действие на 1–2 дня по ПРИМЕРАМ из данных (прочитать конкретную
  статью, запустить конкретный репозиторий, написать обзор, собрать прототип);
kill_criteria — признак провала: что НЕ должно случиться за 4–8 недель, чтобы сигнал стоило
  бросить (например: «за 6 недель не появилось новых статей и репозиториев с этим термином»);
earn_why и earn (0–10) — можно ли за год одному человеку с ИИ-инструментами получить
  с этого продукт, услугу или доход. Оценивай для любого человека, без фантазий.
  Фундаментальная наука (космос, биология, физика) без прямого применения — 0–3;
  инструмент, который уже можно скачать и встроить в продукт, — 6–10;
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
    def _filter_prompt(term: str, titles: str) -> str:
        return f"Термин: «{term}»\nПримеры работ: {titles}"

    def filter(self, cands: list[dict]) -> None:
        shots = [(self._filter_prompt(t, ex), a) for t, ex, a in FILTER_EXAMPLES]
        for c in cands:
            titles = " | ".join(e["title"][:110] for e in c["examples"][:6])
            user = (self._filter_prompt(c["label"], titles) +
                    f"\nИсточники за неделю: " +
                    ", ".join(f"{SOURCE_LABEL[s]} {n}" for s, n in c["by_source"].items()))
            t = time.time()
            r = self._ask(FILTER_SYSTEM, user, FILTER_SCHEMA, 120, shots)
            log.info("фильтр «%s» → %s / %s (%.0fс)", c["label"], r and r.get("kind"),
                     r and r.get("age"), time.time() - t)
            if not r:
                continue
            c["llm_title"] = (r.get("title_ru") or "").strip()
            c["llm_what"] = r.get("what")
            new = str(r.get("age", "")).startswith("новое")
            if r.get("kind") not in KINDS_KEEP or not new:
                c["keep"] = False
                c["reject_reason"] = f"{r.get('kind')}, {'новое' if new else 'известно давно'}: {r.get('what')}"

    # ---------------------------------------------------------------- разбор
    def write(self, c: dict, profile: str) -> dict | None:
        toy = c.get("toy") or {}
        data = {
            "термин": c["label"], "название": c.get("llm_title") or c["label"],
            "упоминаний_за_неделю": c["now"], "ожидалось_по_истории": c["expected"],
            "рост_в_разах": round(2 ** c["growth"], 1),
            "по_источникам": {SOURCE_LABEL[s]: n for s, n in c["by_source"].items()},
            "звёзд_GitHub_в_сутки": c.get("stars_per_day") or None,
            "статей_в_новостях_за_4_недели": c.get("media"),
            "HN_скептиков_и_энтузиастов": [toy.get("skeptic"), toy.get("enthusiast")] if toy else None,
            "примеры": [f"[{SOURCE_LABEL[e['source']]}] {e['title'][:150]}"
                        + (f" — {e['snippet'][:300]}" if e.get("snippet") else "")
                        for e in c["examples"][:6]],
        }
        # профиль пользователя маленькой модели не даём: она начинает притягивать к нему
        # всё подряд. Связь с интересами считаем в коде (поле fit).
        user = f"Данные сигнала:\n{json.dumps(data, ensure_ascii=False, indent=1)}"
        t = time.time()
        r = self._ask(WRITE_SYSTEM, user, WRITE_SCHEMA, 600)
        log.info("разбор «%s» (%.0fс)", c["label"], time.time() - t)
        return r

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
    c["analysis"] = {k: r.get(k) for k in ("what", "why_now", "earn_why", "learn_why",
                                            "cheap_bet", "kill_criteria")}
    c["analysis"]["edge_for_user"] = edge_text(c)
    c["analysis"].update({"title_ru": c.get("llm_title") or c["label"],
                          "skeptics_vs_fans": None, "by": "local"})
    for k in ("earn", "learn"):
        if isinstance(r.get(k), (int, float)):
            c[k] = float(max(0, min(10, r[k])))
