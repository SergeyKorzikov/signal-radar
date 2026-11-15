"""Общие структуры данных."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

# Три «типа» источников. Сигнал сильнее, если виден сразу в нескольких типах.
RESEARCH = "research"  # статьи: arXiv, bioRxiv
BUILDER = "builder"    # код и модели: GitHub, Hugging Face
TALK = "talk"          # обсуждения: Hacker News

SOURCE_KIND = {
    "arxiv": RESEARCH,
    "biorxiv": RESEARCH,
    "github": BUILDER,
    "huggingface": BUILDER,
    "hn": TALK,
}

SOURCE_LABEL = {
    "arxiv": "arXiv",
    "biorxiv": "bioRxiv",
    "github": "GitHub",
    "huggingface": "Hugging Face",
    "hn": "Hacker News",
    "wikipedia": "Википедия", "trends": "Google Trends", "kym": "Know Your Meme",
    "urban": "Urban Dictionary", "roblox": "Roblox", "modrinth": "Modrinth",
    "steam": "Steam", "itch": "itch.io", "polymarket": "Polymarket", "coingecko": "CoinGecko",
}


@dataclass
class Doc:
    source: str          # arxiv | biorxiv | github | huggingface | hn
    id: str
    title: str
    text: str            # всё, из чего извлекаем термины (заголовок + аннотация/описание/теги)
    url: str
    weight: float = 1.0  # вес документа (звёзды, очки и т.п., уже сжатые логарифмом)
    domain: str = ""     # предметная область (категория arXiv/bioRxiv, тема репозитория…)
    tags: list[str] = field(default_factory=list)  # готовые теги (topics GitHub, теги HF)
    meta: dict = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return SOURCE_KIND[self.source]


@dataclass
class Window:
    """Неделя наблюдения: [start, end)."""
    start: datetime
    end: datetime

    @property
    def week_id(self) -> str:
        last_day = (self.end - timedelta(seconds=1)).date()
        y, w, _ = last_day.isocalendar()
        return f"{y}-W{w:02d}"

    @classmethod
    def ending(cls, end: datetime) -> "Window":
        return cls(end - timedelta(days=7), end)

    @classmethod
    def last_full_week(cls, now: datetime | None = None, lag_days: int = 2) -> "Window":
        """7 полных суток (UTC), закончившихся lag_days назад.

        Отступ нужен, потому что arXiv публикует статьи с задержкой в 1–2 дня:
        без него последняя неделя всегда «неполная» и перекошена по составу."""
        now = now or datetime.now(timezone.utc)
        end = datetime(now.year, now.month, now.day, tzinfo=timezone.utc) - timedelta(days=lag_days)
        return cls.ending(end)

    def previous(self, n: int = 1) -> "Window":
        return Window(self.start - timedelta(days=7 * n), self.end - timedelta(days=7 * n))


def to_date(d: datetime) -> date:
    return d.date()
