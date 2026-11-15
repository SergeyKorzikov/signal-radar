"""HTTP с повторами и вежливыми паузами (у arXiv и GDELT строгие лимиты)."""
from __future__ import annotations

import logging
import time

import requests

log = logging.getLogger("radar.http")

_session = requests.Session()
_session.headers["User-Agent"] = "weak-signal-radar/1.0 (+https://github.com)"
_last_call: dict[str, float] = {}


def get(url: str, *, params=None, headers=None, host_delay: float = 0.0,
        retries: int = 4, timeout: float = 60.0, ok_statuses=(200,)) -> requests.Response | None:
    """GET с паузой между запросами к одному хосту и экспоненциальными повторами.

    Возвращает None, если источник так и не ответил: радар не должен падать
    из-за одного недоступного API.
    """
    host = url.split("/")[2]
    for attempt in range(retries):
        wait = _last_call.get(host, 0) + host_delay - time.time()
        if wait > 0:
            time.sleep(wait)
        try:
            r = _session.get(url, params=params, headers=headers, timeout=timeout)
            _last_call[host] = time.time()
            if r.status_code in ok_statuses:
                return r
            if r.status_code in (403, 429, 500, 502, 503, 504):
                backoff = max(host_delay, 2.0) * (2 ** attempt)
                log.warning("%s -> %s, повтор через %.0fс", host, r.status_code, backoff)
                time.sleep(backoff)
                continue
            log.warning("%s -> %s: %s", host, r.status_code, r.text[:200])
            return None
        except requests.RequestException as e:
            _last_call[host] = time.time()
            log.warning("%s: %s (попытка %d)", host, e, attempt + 1)
            time.sleep(2 ** attempt)
    return None
