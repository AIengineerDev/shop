"""Command line entry point: python -m catalog_onboard ingest --db path.sqlite [...]"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from catalog_onboard.config import Settings
from catalog_onboard.db.store import Store, open_readwrite
from catalog_onboard.pipeline.engine import Engine, ItemOutcome
from catalog_onboard.pipeline.llm import LLMClient
from catalog_onboard.pipeline.llm.fake import FakeLLM
from catalog_onboard.rules import load_rules
from catalog_onboard.taxonomy import Taxonomy, TaxonomyConfigError

EXIT_OK = 0
EXIT_CONFIG = 2


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for k in ("feed_item_id", "run_id"):
            if hasattr(record, k):
                payload[k] = getattr(record, k)
        return json.dumps(payload)


def _configure_logging(verbose: bool) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter())
    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING, handlers=[handler])


def make_llm(settings: Settings) -> LLMClient:
    if settings.llm_provider == "fake":
        return FakeLLM()
    from catalog_onboard.pipeline.llm.anthropic_client import AnthropicClient

    return AnthropicClient(model=settings.llm_model, max_tokens=settings.llm_max_tokens)


def cmd_ingest(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    if args.llm:
        settings = Settings(**{**settings.__dict__, "llm_provider": args.llm})
    db = Path(args.db)
    if not db.exists():
        print(f"error: database not found: {db}", file=sys.stderr)
        return EXIT_CONFIG
    try:
        rules = load_rules(settings.rules_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: rules: {exc}", file=sys.stderr)
        return EXIT_CONFIG

    store = Store(open_readwrite(db))
    if not store.has_starter_tables():
        print("error: database lacks canonical_categories / incoming_supplier_feeds", file=sys.stderr)
        return EXIT_CONFIG
    store.migrate()
    try:
        taxonomy = Taxonomy.from_store(store)
    except TaxonomyConfigError as exc:
        print(f"error: taxonomy: {exc}", file=sys.stderr)
        return EXIT_CONFIG

    engine = Engine(
        store=store, taxonomy=taxonomy, rules=rules, llm=make_llm(settings), settings=settings, dry_run=args.dry_run
    )

    def report(o: ItemOutcome) -> None:
        if args.verbose and not o.skipped:
            codes = ",".join(c.code for c in o.reason_codes)
            print(f"{o.feed_item_id}\t{o.status}\t{codes}", file=sys.stderr)

    summary = engine.run_batch(supplier_id=args.supplier, limit=args.limit, on_item=report)
    rows = [
        ("run_id", summary.run_id),
        ("rules_version", rules.version),
        ("llm", settings.llm_provider),
        ("dry_run", str(args.dry_run)),
        ("processed", str(summary.processed)),
        ("draft", str(summary.draft)),
        ("quarantined", str(summary.quarantined)),
        ("errors", str(summary.errors)),
        ("skipped (already processed)", str(summary.skipped)),
    ]
    width = max(len(k) for k, _ in rows)
    for k, v in rows:
        print(f"{k.ljust(width)}  {v}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="catalog_onboard")
    sub = p.add_subparsers(dest="cmd", required=True)
    ing = sub.add_parser("ingest", help="process incoming_supplier_feeds into draft_catalog / quarantine")
    ing.add_argument("--db", required=True)
    ing.add_argument("--limit", type=int, default=None)
    ing.add_argument("--supplier", default=None)
    ing.add_argument("--dry-run", action="store_true")
    ing.add_argument("--llm", choices=["anthropic", "fake"], default=None)
    ing.add_argument("-v", "--verbose", action="store_true")
    ing.set_defaults(func=cmd_ingest)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(getattr(args, "verbose", False))
    return int(args.func(args))
