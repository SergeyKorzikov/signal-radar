"""Hugging Face: модели и Spaces, которые сейчас в тренде.

Исторических срезов у HF нет, поэтому этот источник работает только на текущую неделю:
он участвует в разнообразии источников и примерах, но не в расчёте роста.
"""
from __future__ import annotations

import logging
import math
import re

from ..http import get
from ..models import Doc, Window

log = logging.getLogger("radar.hf")

# теги-«шум», не несущие смысла
_SKIP = re.compile(r"^(license:|region:|base_model:|arxiv:|dataset:|doi:|endpoints|"
                   r"autotrain|safetensors|gguf|transformers|pytorch|text-generation-inference|"
                   r"[a-z]{2}$)")


def _clean_tags(tags: list[str]) -> list[str]:
    return [t.replace("-", " ").replace("_", " ") for t in tags if not _SKIP.match(t)]


def fetch(window: Window, limit: int = 300) -> list[Doc]:
    docs: list[Doc] = []
    for kind in ("models", "spaces"):
        r = get(f"https://huggingface.co/api/{kind}",
                params={"sort": "trendingScore", "limit": limit}, host_delay=0.5)
        if r is None:
            continue
        for it in r.json():
            rid = it.get("id") or it.get("modelId") or ""
            name = rid.split("/")[-1]
            tags = _clean_tags(it.get("tags") or [])
            pipeline = it.get("pipeline_tag") or ""
            likes = it.get("likes", 0)
            title = f"{rid} ({pipeline})" if pipeline else rid
            text = f"{re.sub(r'[-_.]', ' ', name)}. {pipeline.replace('-', ' ')}. {' '.join(tags)}"
            url = f"https://huggingface.co/{'spaces/' if kind == 'spaces' else ''}{rid}"
            docs.append(Doc(source="huggingface", id=rid, title=title, text=text, url=url,
                            weight=1.0 + math.log1p(it.get("trendingScore", 0) or 0) / 2,
                            domain=pipeline, tags=tags, meta={"likes": likes, "kind": kind}))
    log.info("Hugging Face: %d", len(docs))
    return docs
