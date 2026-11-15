"""Хранилище недельных частот в репозитории: data/weeks/<неделя>.json.gz."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEEKS = ROOT / "data" / "weeks"
DIGESTS = ROOT / "data" / "digests"
KEEP_FULL_WEEKS = 12      # сколько недель хранить в полной детализации
PRUNED_MIN_DF = 8         # что оставлять в более старых неделях


def save_week(week_id: str, payload: dict, root: Path = WEEKS) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    p = root / f"{week_id}.json.gz"
    with gzip.open(p, "wt", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    return p


def load_week(week_id: str, root: Path = WEEKS) -> dict | None:
    p = root / f"{week_id}.json.gz"
    if not p.exists():
        return None
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return json.load(f)


def list_weeks(root: Path = WEEKS) -> list[str]:
    return sorted(p.name.removesuffix(".json.gz") for p in root.glob("*.json.gz"))


def compact(keep_terms: set[str], root: Path = WEEKS) -> None:
    """Старые недели ужимаем: оставляем частые термины и те, что уже были сигналами."""
    weeks = list_weeks(root)
    for w in weeks[:-KEEP_FULL_WEEKS]:
        data = load_week(w, root)
        if not data or data.get("compacted"):
            continue
        data["terms"] = {t: c for t, c in data["terms"].items()
                         if t in keep_terms or sum(c.values()) >= PRUNED_MIN_DF}
        data["compacted"] = True
        save_week(w, data, root)


def save_digest(week_id: str, digest: dict, root: Path = DIGESTS) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    p = root / f"{week_id}.json"
    p.write_text(json.dumps(digest, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def list_digests(root: Path = DIGESTS) -> list[str]:
    return sorted(p.stem for p in root.glob("*.json"))


def load_digest(week_id: str, root: Path = DIGESTS) -> dict | None:
    p = root / f"{week_id}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
