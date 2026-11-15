"""Командная строка радара.

  python -m radar run                 # прогон радара (сам догрузит историю при первом запуске)
  python -m radar backfill --weeks 8  # догрузить историю
  python -m radar build --out site    # собрать дашборд из data/
  python -m radar backtest            # проверить алгоритм на прошлых волнах
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from . import backtest, pipeline, site


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="radar")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--no-backfill", action="store_true")
    b = sub.add_parser("backfill")
    b.add_argument("--weeks", type=int, default=8)
    s = sub.add_parser("build")
    s.add_argument("--out", default="site")
    sub.add_parser("backtest")
    c = sub.add_parser("chat")
    c.add_argument("--issue", type=int, required=True)
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stderr)
    cfg = pipeline.load_config()
    if a.cmd == "run":
        d = pipeline.run(cfg, auto_backfill=0 if a.no_backfill else 6)
        print(f"Неделя {d['week']}: {len(d['main'])} сигналов + {len(d['outside'])} вне пузыря "
              f"(разбор: {d['analyst']})")
    elif a.cmd == "backfill":
        pipeline.backfill(a.weeks, cfg)
    elif a.cmd == "build":
        out = site.build(a.out)
        print(f"Дашборд: {out}")
    elif a.cmd == "chat":
        from . import chat
        print(chat.answer_issue(a.issue, cfg)[:300])
    elif a.cmd == "backtest":
        res = backtest.run(cfg["backtest"])
        out = pipeline.ROOT / "data" / "backtest.json"
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        for x in res:
            lead = f"за {x['lead_months']} мес. до хайпа" if x["lead_months"] is not None else "не поймал"
            print(f"{x['label']:28} сигнал: {x['fired'] or '—':8} хайп: {x['hype']}  → {lead}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
