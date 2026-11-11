"""CLI: python -m repricer <command>."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from repricer.clock import utcnow
from repricer.config import Settings
from repricer.db import repo
from repricer.db.connection import connect, table_exists
from repricer.db.migrate import migrate
from repricer.engine.policy import PolicyStore
from repricer.execution import audit, review, runner
from repricer.extraction.pipeline import extract_stage
from repricer.money import fmt
from repricer.seed import generate, write_fixtures, write_legacy_db


def _settings(args: argparse.Namespace) -> Settings:
    s = Settings.from_env()
    if args.db:
        s = Settings(**{**s.__dict__, "db_path": Path(args.db)})
    return s


def _extractors(s: Settings) -> Any:
    from repricer.api.app import build_extractors

    return build_extractors(s)


def cmd_migrate(args: argparse.Namespace) -> int:
    s = _settings(args)
    conn = connect(s.db_path)
    rep = migrate(conn)
    print(
        f"migrated={rep.migrated_legacy} inventory={rep.inventory_rows} "
        f"scrapes={rep.scrape_rows} applied={list(rep.applied_versions)}"
    )
    return 0


def cmd_seed(args: argparse.Namespace) -> int:
    s = _settings(args)
    conn = connect(s.db_path)
    if table_exists(conn, "inventory") and not args.force:
        print("inventory already exists; use --force to drop and reseed", file=sys.stderr)
        return 2
    if args.force:
        conn.close()
        for suffix in ("", "-wal", "-shm"):
            p = Path(str(s.db_path) + suffix)
            if p.exists():
                p.unlink()
        conn = connect(s.db_path)
    data = generate(seed=args.seed, n_skus=args.skus, n_scrapes=args.scrapes)
    write_legacy_db(conn, data)
    print(f"seeded legacy schema: {len(data.inventory)} skus, {len(data.scrapes)} scrapes")
    if args.fixtures:
        n = write_fixtures(Path(args.fixtures), data)
        print(f"wrote {n} golden fixtures to {args.fixtures}")
    if not args.no_migrate:
        rep = migrate(conn)
        print(f"migrated to cents: inventory={rep.inventory_rows} scrapes={rep.scrape_rows}")
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    s = _settings(args)
    conn = connect(s.db_path)
    primary, fallback = _extractors(s)
    summary = extract_stage(
        conn,
        extractor=primary,
        now=utcnow(),
        batch_size=s.llm_batch,
        fallback=fallback,
        force=args.force,
    )
    print(json.dumps(summary.__dict__))
    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    s = _settings(args)
    conn = connect(s.db_path)
    store = PolicyStore(s.policies_path)
    res = runner.run(
        conn,
        store=store,
        trigger="cli",
        now=utcnow(),
        hard_cap_bp=s.hard_cap_bp,
        stages=("recommend",),
    )
    print(json.dumps(res.__dict__))
    return 0 if res.status == "completed" else 1


def cmd_run(args: argparse.Namespace) -> int:
    s = _settings(args)
    conn = connect(s.db_path)
    store = PolicyStore(s.policies_path)
    primary, fallback = _extractors(s)
    now = utcnow()

    def extract(c: Any, _rid: str) -> int:
        return extract_stage(
            c, extractor=primary, now=now, batch_size=s.llm_batch, fallback=fallback
        ).scrapes

    res = runner.run(
        conn, store=store, trigger="cli", now=now, hard_cap_bp=s.hard_cap_bp, extract=extract
    )
    print(json.dumps(res.__dict__))
    if res.status == "completed":
        rows, total = repo.list_recommendations(conn, run_id=res.run_id, decision="propose")
        print(f"{total} proposals pending review; e.g.:")
        for r in rows[:5]:
            print(
                f"  {r['recommendation_id']} {r['sku']} {fmt(r['current_price_cents'])} -> "
                f"{fmt(r['proposed_price_cents'])}  {json.loads(r['reason_codes'])}"
            )
    return 0 if res.status == "completed" else 1


def cmd_apply(args: argparse.Namespace) -> int:
    s = _settings(args)
    conn = connect(s.db_path)
    change_id = review.approve(
        conn,
        args.recommendation_id,
        actor=args.actor,
        note=args.note,
        now=utcnow(),
        hard_cap_bp=s.hard_cap_bp,
    )
    print(f"applied change {change_id}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    s = _settings(args)
    conn = connect(s.db_path)
    if args.what == "verify":
        breaks = audit.verify_chain(conn)
        if breaks:
            for b in breaks:
                print(f"BREAK {b.sku} {b.change_id}: {b.reason}")
            return 1
        print("chain ok")
        return 0
    if args.what == "sku":
        for row in audit.history(conn, args.id):
            print(
                f"{row['applied_at']} {row['sku']} {fmt(row['old_price_cents'])} -> "
                f"{fmt(row['new_price_cents'])} [{row['source']}] by {row['actor']} "
                f"{row.get('reason_codes', '')}"
            )
        return 0
    report = audit.run_report(conn, args.id)
    for item in report["items"]:
        print(
            f"{item['recommendation_id']} {item['sku']} {item['decision']:8} "
            f"{fmt(item['current_price_cents'])} -> {fmt(item['proposed_price_cents'])} "
            f"{item['reason_codes']}"
        )
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from repricer.api.app import create_app

    uvicorn.run(create_app(_settings(args)), host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="repricer")
    p.add_argument("--db", help="SQLite path (overrides REPRICER_DB_PATH)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("migrate", help="apply schema; convert a legacy REAL-money DB").set_defaults(
        fn=cmd_migrate
    )
    sp = sub.add_parser("seed", help="write deterministic synthetic data (then migrate)")
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--skus", type=int, default=50)
    sp.add_argument("--scrapes", type=int, default=200)
    sp.add_argument("--fixtures", help="also write golden fixtures to this directory")
    sp.add_argument("--no-migrate", action="store_true")
    sp.add_argument("--force", action="store_true", help="delete the DB file first")
    sp.set_defaults(fn=cmd_seed)
    sp = sub.add_parser("extract", help="extraction stage only")
    sp.add_argument("--force", action="store_true", help="re-extract everything (supersede)")
    sp.set_defaults(fn=cmd_extract)
    sub.add_parser("recommend", help="recommendation stage only").set_defaults(fn=cmd_recommend)
    sub.add_parser("run", help="extract + recommend in one run").set_defaults(fn=cmd_run)
    sp = sub.add_parser("apply", help="approve + apply a pending recommendation")
    sp.add_argument("recommendation_id")
    sp.add_argument("--actor", default="cli")
    sp.add_argument("--note")
    sp.set_defaults(fn=cmd_apply)
    sp = sub.add_parser("audit")
    sp.add_argument("what", choices=["verify", "sku", "run"])
    sp.add_argument("id", nargs="?")
    sp.set_defaults(fn=cmd_audit)
    sp = sub.add_parser("serve")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)
    sp.set_defaults(fn=cmd_serve)

    args = p.parse_args(argv)
    if args.cmd == "audit" and args.what in ("sku", "run") and not args.id:
        p.error(f"audit {args.what} needs an id")
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
