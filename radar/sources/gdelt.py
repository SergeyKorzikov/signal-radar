"""GDELT: сколько о термине пишут мировые СМИ (мерило «мейнстримности»)."""
from __future__ import annotations

import logging

from ..http import get

log = logging.getLogger("radar.gdelt")
API = "https://api.gdeltproject.org/api/v2/doc/doc"


def media_count(term: str, timespan: str = "4w") -> int | None:
    """Число статей в новостях за период. None — если GDELT не ответил."""
    r = get(API, params={"query": f'"{term}"', "mode": "timelinevolraw",
                         "timespan": timespan, "format": "json"},
            host_delay=6.0, retries=2, timeout=25)
    if r is None:
        return None
    try:
        data = r.json()
    except ValueError:
        # на слишком короткие/частые запросы GDELT отвечает текстом, а не JSON
        return None
    total = 0
    for series in data.get("timeline", []):
        for pt in series.get("data", []):
            total += int(pt.get("value", 0))
    return total
