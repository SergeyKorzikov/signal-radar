"""Сборка дашборда: одна статическая страница для GitHub Pages."""
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

from . import history
from .models import SOURCE_LABEL

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "templates" / "dashboard.html"


def _read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def collect_data() -> dict:
    digests = [history.load_digest(w) for w in history.list_digests()[-12:]]
    for d in digests:  # в публичную страницу не попадает ничего личного
        for c in (d or {}).get("main", []) + (d or {}).get("outside", []):
            (c.get("analysis") or {}).pop("edge_for_user", None)
            c.pop("fit", None)
            c.pop("fit_domain", None)
    tracker_p = ROOT / "tracker.yaml"
    tracker = {}
    if tracker_p.exists():
        raw = yaml.safe_load(tracker_p.read_text(encoding="utf-8")) or {}
        tracker = {str(k).lower(): v for k, v in (raw.get("signals") or {}).items()}
    backtest = _read_json(ROOT / "data" / "backtest.json")
    if backtest:
        for b in backtest:  # в дашборд — только нужное
            b["rows"] = [{"m": r["month"], "rate": r["rate"]} for r in b["rows"]]
    return {"digests": [d for d in digests if d][::-1],  # новые сначала
            "registry": _read_json(ROOT / "data" / "signals.json") or {},
            "tracker": tracker, "backtest": backtest, "sources": SOURCE_LABEL,
            "repo": os.environ.get("GITHUB_REPOSITORY", "")}


def build(out: str | Path = "site") -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    data = json.dumps(collect_data(), ensure_ascii=False).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__RADAR_DATA__*/null", data)
    p = out / "index.html"
    p.write_text(html, encoding="utf-8")
    (out / ".nojekyll").write_text("")
    return p
